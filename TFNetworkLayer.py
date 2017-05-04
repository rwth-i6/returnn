
from __future__ import print_function

import tensorflow as tf
import TFUtil
from TFUtil import Data, OutputWithActivation, reuse_name_scope, var_creation_scope, dimshuffle, swapaxes


class LayerBase(object):
  layer_class = None
  recurrent = False

  def __init__(self, name, network, output=None, n_out=None, out_type=None, sources=(),
               target=None, loss=None, size_target=None,
               L2=None, is_output_layer=None,
               batch_norm=False,
               spatial_smoothing=0.0,
               initial_output=None,
               rec_previous_layer=None,
               trainable=True):
    """
    :param str name:
    :param TFNetwork.TFNetwork network:
    :param Data output:
    :param None|int n_out: output dim
    :param dict[str] out_type: kwargs for Data class. more explicit than n_out.
    :param list[LayerBase] sources: via self.transform_config_dict()
    :param str|None target: if some loss is set, this is the target data-key, i.e. network.extern_data.get_data(target)
      alternatively, this also can be a layer name.
    :param str|None size_target: like target but this is only used to set our output size in case of training
    :param Loss|None loss: via self.transform_config_dict()
    :param float|None L2: for constraints
    :param bool|None is_output_layer:
    :param bool|dict batch_norm:
    :param str|float initial_output: used for recurrent layer, see self.get_rec_initial_output()
    :param LayerBase|None rec_previous_layer: via the recurrent layer, layer (template) which represents the past of us
    :param bool trainable: whether the parameters of this layer will be trained
    """
    self.name = name
    self.network = network
    if loss and not target:
      target = self.network.extern_data.default_target
    self.target = target
    self.loss = loss
    if self.loss and self.loss.recurrent:
      self.recurrent = True
    if output:
      self.output = output
      if n_out:
        assert self.output.dim == n_out
      if out_type:
        if "shape" in out_type:
          assert self.output.shape == out_type["shape"]
        if "dim" in out_type:
          assert self.output.dim == out_type["dim"]
    else:
      self.output = self.get_out_data_from_opts(
        out_type=out_type, n_out=n_out,
        network=network, name=name, target=target, size_target=size_target,
        sources=sources, loss=loss)
    self.output_before_activation = None  # type: None|OutputWithActivation
    self.rec_vars_outputs = {}  # type: dict[str,tf.Tensor]
    self._initial_output = initial_output
    self._rec_previous_layer = rec_previous_layer
    self.sources = sources
    self.params = {}  # type: dict[str,tf.Variable]
    self.L2 = L2
    self._is_output_layer = is_output_layer
    self.use_batch_norm = batch_norm
    self.spatial_smoothing = spatial_smoothing
    self.trainable = trainable
    # Stats will be collected by the engine.
    self.stats = {}  # type: dict[str,tf.Tensor]

  def post_init(self):
    """
    This gets called right after self.__init__().
    """
    if self.use_batch_norm:
      opts = {}
      if isinstance(self.use_batch_norm, dict):
        opts = self.use_batch_norm
      self.output.placeholder = self.batch_norm(self.output, **opts)

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
  def _base_get_out_data_from_opts(cls, network, name, out_type=None, n_out=None, target=None, size_target=None,
                                   sources=(), loss=None,
                                   **kwargs):
    """
    Called via BaseLayer.get_out_data_from_opts().

    :param TFNetwork.TFNetwork network:
    :param str name:
    :param dict[str]|None out_type:
    :param int|None n_out:
    :param str|None target:
    :param str|None size_target:
    :param list[LayerBase] sources:
    :param Loss|None loss:
    :param kwargs: remaining kwargs of self.__init__(), ignored here
    :return: Data template (placeholder not set)
    :rtype: Data
    """
    if loss and not target:
      target = network.extern_data.default_target
    if out_type is None and n_out is None and target:
      n_out = cls._static_get_target_value(target=target, network=network, mark_data_key_as_used=False).dim
      if loss:
        n_out = loss.get_auto_output_layer_dim(n_out)
    if out_type is None:
      assert n_out
      out_type = {"dim": n_out}
    out_type = out_type.copy()
    out_type.setdefault("name", "%s_output" % name)
    if sources and not sources[0].output.sparse:
      out_type.setdefault("dtype", sources[0].output.dtype)
    if n_out is not None:
      out_type.setdefault("dim", n_out)
      assert out_type["dim"] == n_out
    if sources:
      if out_type.get("sparse", False):
        out_type.setdefault("shape", sources[0].output.shape_dense[:-1])
      else:
        out_type.setdefault("shape", sources[0].output.shape_dense[:-1] + (out_type.get("dim"),))
    # You are supposed to set self.output.{batch_dim_axis,time_dim_axis} explicitly,
    # as well as check the inputs if they are as you would suggest.
    # However, a good default is often to use the same as the input.
    if sources and "batch_dim_axis" not in out_type:
      out_type.setdefault("batch_dim_axis", sources[0].output.batch_dim_axis)
      out_type.setdefault("time_dim_axis", sources[0].output.time_dim_axis)
    output = Data(**out_type)
    # You are supposed to set self.output.placeholder to the value which you want to return by the layer.
    # Normally you are also supposed to set self.output.size_placeholder explicitly, just like self.output.placeholder.
    # However, in many cases, this will just be {0: time-lengths} and the same as from the input.
    # We check for this case and preset it by that if possible.
    # If you want to have it different in your layer, just overwrite it.
    if sources and sources[0].output.matches_var_dim_pattern(output):
      output.size_placeholder = sources[0].output.size_placeholder.copy()
    elif target or size_target:
      # TODO: In training, this is ok. Maybe as well as for eval but not clear.
      # In forward, mark_data_key_as_used=False should be used and anyway that target value is not available.
      output.size_placeholder = cls._static_get_target_value(
        target=target or size_target, network=network,
        mark_data_key_as_used=network.train_flag is not False).size_placeholder.copy()
    return output

  def __repr__(self):
    return "%s{class=%s, out_type=%s}" % (
      self.name, self.layer_class, self.output.get_description(with_name=False))

  @classmethod
  def cls_get_tf_scope_name(cls, name):
    """
    :param str name: layer name
    :return: scope name, might be just name
    """
    return name.replace(":", "__")

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param TFNetwork.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer

    Will modify `d` such that it becomes the kwargs for `self.__init__()`.
    Mostly leaves `d` as-is.
    This is used by TFNetwork.construct_from_dict().
    """
    src_names = d.pop("from", ["data"])
    if not isinstance(src_names, (list, tuple)):
      src_names = [src_names]
    d["sources"] = [
      get_layer(src_name)
      for src_name in src_names
      if not src_name == "none"]
    if d.get("loss"):
      loss_class = get_loss_class(d["loss"])
      d["loss"] = loss_class(**d.pop("loss_opts", {}))
    if d.get("target"):
      # Not resolving this in the dict, but call get_layer to make it available.
      assert isinstance(d["target"], str)
      if d["target"].startswith("layer:"):
        get_layer(d["target"][len("layer:"):])

  @property
  def tf_scope_name(self):
    return self.cls_get_tf_scope_name(name=self.name)

  def is_output_layer(self):
    """
    Some code differs between an output layer and other layers.
    It is a bit arbitrary what we define as output layer.
    :rtype: bool
    """
    if self._is_output_layer is not None:
      return self._is_output_layer
    if self.target:
      return True
    if self.name == "output":
      return True
    return False

  def add_param(self, param):
    """
    :param tf.Variable param:
    :return: param
    :rtype tf.Variable
    """
    assert param.name
    self.params[param.name] = param
    return param

  def set_param_values_by_dict(self, values_dict, session):
    """
    :param dict[str,numpy.ndarray] values_dict:
    :param tf.Session session:
    """
    for param_name, values in values_dict.items():
      param = self.params[param_name]
      assert isinstance(param, tf.Variable)
      shape = param.get_shape()
      assert isinstance(shape, tf.TensorShape)
      assert shape.is_fully_defined()
      assert tuple(shape.as_list()) == values.shape
      self.network.get_var_assigner(param).assign(values, session=session)

  def get_param_values_dict(self, session):
    """
    :param tf.Session session:
    :return: dict name -> values
    :rtype: dict[str,numpy.ndarray]
    """
    d = {}
    for param_name, param in self.params.items():
      d[param_name] = param.eval(session)
    return d

  @staticmethod
  def _static_get_target_value(target, network, mark_data_key_as_used=True):
    """
    :param str target:
    :param TFNetwork.TFNetwork network:
    :param bool mark_data_key_as_used: forwarded self.network.get_extern_data()
    :rtype: Data | None
    """
    if not target or target == "none":
      return None
    if target.startswith("layer:"):
      return network.layers[target[len("layer:"):]].output
    assert network.extern_data.has_data(target), "target %r unknown" % target
    return network.get_extern_data(target, mark_data_key_as_used=mark_data_key_as_used)

  def _get_target_value(self, mark_data_key_as_used=True):
    """
    :param bool mark_data_key_as_used: forwarded self.network.get_extern_data()
    :rtype: Data | None
    """
    return self._static_get_target_value(
      target=self.target, network=self.network, mark_data_key_as_used=mark_data_key_as_used)

  def _init_loss(self):
    if self.loss.output is self.output:
      return
    self.loss.init(
      output=self.output,
      output_with_activation=self.output_before_activation,
      target=self._get_target_value())

  def get_loss_value(self):
    """
    :return: the loss, a scalar value, or None if not set
    :rtype: tf.Tensor | None
    """
    if not self.loss:
      return None
    self._init_loss()
    with tf.name_scope("loss"):
      return self.loss.get_value()

  def get_error_value(self):
    """
    :return: usually the frame error rate, or None if not defined
    :rtype: tf.Tensor | None
    """
    if not self.loss:
      return None
    self._init_loss()
    with tf.name_scope("error"):
      return self.loss.get_error()

  def get_loss_normalization_factor(self):
    if not self.loss:
      return None
    self._init_loss()
    return self.loss.get_normalization_factor()

  def get_params_l2_norm(self):
    return 2 * sum([tf.nn.l2_loss(param) for (name, param) in sorted(self.params.items())])

  def get_output_spatial_smoothing_energy(self):
    from TFUtil import spatial_smoothing_energy, flatten_with_seq_len_mask
    energy = spatial_smoothing_energy(self.output.placeholder, dim=self.output.dim)  # (batch,time)
    assert self.output.have_tim_axis()
    energy = flatten_with_seq_len_mask(
      energy,
      seq_lens=self.output.size_placeholder[self.output.time_dim_axis_excluding_batch],
      time_major=self.output.is_time_major)  # (time')
    energy = tf.reduce_sum(energy)
    return energy

  def get_constraints_value(self):
    c = 0
    if self.L2:
      c += self.L2 * self.get_params_l2_norm()
    if self.spatial_smoothing:
      c += self.spatial_smoothing * self.get_output_spatial_smoothing_energy()
    if c is 0:
      return None
    return c

  def batch_norm(self, data,
                 use_shift=True, use_std=True, use_sample=0.0, force_sample=False,
                 momentum=0.99, epsilon=1e-3,
                 sample_mean=None, sample_variance=None,
                 gamma=None, beta=None):
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
    :rtype: tf.Tensor

    http://arxiv.org/abs/1502.03167

    Also see:
      tf.nn.batch_normalization()
      https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/batch_norm.py
    """
    with tf.name_scope("batch_norm"):
      x = data.get_placeholder_flattened(keep_dims=True)  # shape (time',...)
      mean, variance = tf.nn.moments(x, axes=[0], keep_dims=True)
      if sample_mean is None:
        with var_creation_scope():
          sample_mean = self.add_param(tf.Variable(
            initial_value=tf.zeros(data.non_dynamic_batch_shape),
            name="%s_%s_mean" % (self.name, data.name),
            trainable=False))
        # Use exponential moving average of batch mean.
        # Note: We could also use cumulative moving average. Our Theano implementation does that for inference.
        sample_mean = tf.assign_add(sample_mean, (mean - sample_mean) * momentum)
      if sample_variance is None:
        # Note: Our Theano implementation does not use a moving average for this.
        with var_creation_scope():
          sample_variance = self.add_param(tf.Variable(
            initial_value=tf.ones(data.non_dynamic_batch_shape),
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
          with var_creation_scope():
            gamma = self.add_param(tf.Variable(
              initial_value=tf.ones(data.non_dynamic_batch_shape),
              name="%s_%s_gamma" % (self.name, data.name),
              trainable=True))
        bn *= gamma
      if use_shift:
        if beta is None:
          with var_creation_scope():
            beta = self.add_param(tf.Variable(
              initial_value=tf.zeros(data.non_dynamic_batch_shape),
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

  def get_last_hidden_state(self):
    """
    If this is a recurrent layer, this would return the last hidden state.
    If not, as a fallback, we recursively check our sources.
    :rtype: tf.Tensor | None
    :return: optional tensor with shape (batch, dim)
    """
    # This is the generic fallback code.
    hidden_states = []
    for s in self.sources:
      h = s.get_last_hidden_state()
      if h is not None:
        assert h.get_shape().ndims == 2
        hidden_states += [h]
    if not hidden_states:
      return None
    if len(hidden_states) == 1:
      return hidden_states[0]
    return tf.concat(hidden_states, axis=1, name="concat_hidden_states")

  @classmethod
  def get_rec_initial_output(cls, batch_dim, name, output, initial_output=None, **kwargs):
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
    assert all([d is not None for d in data.shape])
    shape = [batch_dim] + list(data.shape)
    if isinstance(v, (float, int)):
      with tf.name_scope("init_%s_const" % name):
        return tf.ones(shape, dtype=data.dtype) * v
    assert isinstance(v, str)
    if v == "zeros":
      return tf.zeros(shape, dtype=data.dtype, name="init_%s_zeros" % name)
    elif v == "ones":
      return tf.ones(shape, dtype=data.dtype, name="init_%s_ones" % name)
    else:
      raise Exception("invalid initial output type %r for sub-layer %r" % (v, name))

  @classmethod
  def get_rec_initial_extra_outputs(cls, batch_dim, **kwargs):
    return {}


class SourceLayer(LayerBase):
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
    data = network.get_extern_data(data_key, mark_data_key_as_used=True)
    super(SourceLayer, self).__init__(network=network, **kwargs)
    self.output = data

  @classmethod
  def get_out_data_from_opts(cls, network, data_key=None, **kwargs):
    if data_key is None:
      data_key = network.extern_data.default_input
    return network.get_extern_data(data_key, mark_data_key_as_used=False)


def concat_sources(src_layers):
  """
  :param list[LayerBase] src_layers:
  :return: data with placeholders set
  :rtype: Data
  """
  assert src_layers, "need source layers"
  if len(src_layers) == 1:
    return src_layers[0].output
  network = src_layers[0].network
  if (tuple(src_layers), 0.0) in network.concat_sources_dropout_cache:
    return network.concat_sources_dropout_cache[(tuple(src_layers), 0.0)].copy()
  data = get_concat_sources_data_template(src_layers)
  prefix_shape = data.shape[:-1]  # without batch-dim
  for layer in src_layers:
    assert not layer.output.sparse, "sparse concat not supported"
    assert layer.output.dtype == data.dtype, "incompatible dtype with layer %r" % layer
    assert layer.output.time_dim_axis_excluding_batch == data.time_dim_axis_excluding_batch
    shape = layer.output.shape
    assert layer.output.placeholder.get_shape().ndims == len(shape) + 1  # with batch-dim
    assert shape, "source must not be a scalar of layer %r" % layer
    assert shape[:-1] == prefix_shape, "incompatible concat with layer %r" % layer
    assert shape[-1], "source last-dim must be specified of layer %r" % layer
  data.placeholder = tf.concat(
    axis=len(prefix_shape) + 1,  # one more because this is with batch-dim
    values=[layer.output.get_placeholder_with_specific_batch_dim_axis(data.batch_dim_axis) for layer in src_layers])
  data.size_placeholder = src_layers[0].output.size_placeholder.copy()
  network.concat_sources_dropout_cache[(tuple(src_layers), 0.0)] = data.copy()
  return data


def get_concat_sources_data_template(src_layers):
  """
  :param list[LayerBase] src_layers:
  :return: data with no placeholders set
  :rtype: Data
  """
  assert src_layers, "need source layers"
  dim = 0
  for layer in src_layers:
    shape = layer.output.shape
    assert shape[-1], "source last-dim must be specified of layer %r" % layer
    dim += shape[-1]
  data = Data(
    name="concat_sources",
    shape=src_layers[0].output.shape[:-1] + (dim,),
    dim=dim,
    sparse=False,
    batch_dim_axis=src_layers[0].output.batch_dim_axis,
    time_dim_axis=src_layers[0].output.time_dim_axis,
    dtype=src_layers[0].output.dtype)
  return data


def concat_sources_with_opt_dropout(src_layers, dropout=0):
  """
  :param list[LayerBase] src_layers:
  :param float dropout: will be applied if train_flag is set
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
    return data
  if (tuple(src_layers), float(dropout)) in network.concat_sources_dropout_cache:
    return network.concat_sources_dropout_cache[(tuple(src_layers), float(dropout))].copy()
  assert 0.0 < dropout < 1.0
  fn_train = lambda: tf.nn.dropout(
      data.placeholder,
      keep_prob=1 - dropout,
      # noise_shape is like old behavior for now:
      # all dynamic dimensions (batch,time) will use the same dropout-mask broadcasted.
      noise_shape=data.non_dynamic_batch_shape,
      seed=network.random.randint(2 ** 31))
  fn_eval = lambda: data.placeholder
  data.placeholder = network.cond_on_train(fn_train, fn_eval)
  network.concat_sources_dropout_cache[(tuple(src_layers), float(dropout))] = data.copy()
  return data


class _ConcatInputLayer(LayerBase):
  def __init__(self, dropout=0, mask=None, **kwargs):
    """
    :param float dropout: 0.0 means to apply no dropout. dropout will only be applied during training
    :param str|None mask: "dropout" or "unity" or None. this is obsolete and only here for historical reasons
    """
    super(_ConcatInputLayer, self).__init__(**kwargs)
    assert mask in ['dropout', 'unity', None], "invalid mask: %r" % mask
    if mask == "unity":
      assert not dropout
    elif mask == "dropout":
      assert dropout > 0
    self.input_data = None
    if self.sources:
      self.input_data = concat_sources_with_opt_dropout(self.sources, dropout=dropout)


class CopyLayer(_ConcatInputLayer):
  """
  This layer does nothing, it copies its input.
  If multiple sources are provided, they are concatenated in the feature-dim.
  """

  layer_class = "copy"

  def __init__(self, **kwargs):
    super(CopyLayer, self).__init__(**kwargs)
    self.output = self.input_data

  @classmethod
  def get_out_data_from_opts(cls, sources=(), out_type=None, n_out=None, **kwargs):
    if out_type or n_out:
      return super(CopyLayer, cls).get_out_data_from_opts(out_type=out_type, n_out=n_out, sources=sources, **kwargs)
    return get_concat_sources_data_template(sources)


class ActivationLayer(CopyLayer):
  """
  This layer just applies an activation function.
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
  layer_class = "batch_norm"

  def __init__(self, **kwargs):
    kwargs = kwargs.copy()
    import inspect
    batch_norm_kwargs = inspect.getargspec(self.batch_norm).args[1:]  # first is self, ignore
    batch_norm_opts = {key: kwargs.pop(key)
                       for key in batch_norm_kwargs
                       if key in kwargs}
    super(BatchNormLayer, self).__init__(use_batch_norm=batch_norm_opts or True, **kwargs)


class SliceLayer(_ConcatInputLayer):
  layer_class = "slice"

  def __init__(self, axis=None, axis_kind=None,
               slice_start=None, slice_end=None, slice_step=None,
               **kwargs):
    """
    :param int|None axis:
    :param str|None axis_kind: "T" for time, "B" for batch, "F" for feature
    :param int|None slice_start:
    :param int|None slice_end:
    :param int|None slice_step:
    :param int|None n_out:
    """
    super(SliceLayer, self).__init__( **kwargs)
    axis = self._get_axis(axis=axis, axis_kind=axis_kind, input_data=self.input_data)
    dim_slice = slice(slice_start, slice_end, slice_step)
    slices = [slice(None, None)] * axis + [dim_slice]
    axis_wo_batch = self.input_data.get_batch_axis_excluding_batch(axis)
    self.output.size_placeholder = self.input_data.size_placeholder
    if axis == self.input_data.time_dim_axis:
      if slice_start:
        assert slice_start > 0
        self.output.size_placeholder[self.input_data.time_dim_axis_excluding_batch] = \
          tf.maximum(0, self.output.size_placeholder[self.input_data.time_dim_axis_excluding_batch] - slice_start)
      if slice_end:
        assert slice_end > 0
        self.output.size_placeholder[self.input_data.time_dim_axis_excluding_batch] = \
          tf.minimum(
            tf.shape(self.input_data.placeholder)[self.input_data.time_dim_axis] - slice_end,
            self.output.size_placeholder[self.input_data.time_dim_axis_excluding_batch])
      if slice_step:
        self.output.size_placeholder[self.input_data.time_dim_axis_excluding_batch] //= slice_step
    elif axis_wo_batch is not None:
      assert axis_wo_batch not in self.output.size_placeholder
    self.output.placeholder = self.input_data.placeholder[slices]

  @classmethod
  def _get_axis(cls, axis, axis_kind, input_data):
    """
    :param int|None axis:
    :param str|None axis_kind: "T" for time, "B" for batch, "F" for feature
    :param Data input_data:
    :return: axis
    :rtype: int
    """
    if axis is not None:
      assert not axis_kind
      assert 0 <= axis < len(input_data.batch_shape)
    else:
      assert axis_kind
      axis_kind = axis_kind.upper()
      if axis_kind == "T":
        assert input_data.time_dim_axis is not None
        axis = input_data.time_dim_axis
      elif axis_kind == "B":
        assert input_data.batch_dim_axis is not None
        axis = input_data.batch_dim_axis
      elif axis_kind == "F":
        axes = input_data.get_axes(exclude_time=True, exclude_batch=True)
        assert len(axes) == 1
        axis = axes[0]
    return axis

  @classmethod
  def get_out_data_from_opts(
        cls, axis=None, axis_kind=None, sources=(),
        slice_start=None, slice_end=None, slice_step=None, **kwargs):
    input_data = get_concat_sources_data_template(sources)
    axis = cls._get_axis(axis=axis, axis_kind=axis_kind, input_data=input_data)
    out_type = input_data.get_kwargs()
    axis_wo_batch = input_data.get_batch_axis_excluding_batch(axis)
    dim_slice = slice(slice_start, slice_end, slice_step)
    if axis_wo_batch is not None:
      out_type["shape"] = list(out_type["shape"])
      if out_type["shape"][axis_wo_batch] is not None:
        out_type["shape"][axis_wo_batch] = len(range(out_type["shape"][axis_wo_batch])[dim_slice])
      if axis_wo_batch == len(out_type["shape"]) - 1 and not out_type["sparse"]:
        out_type["dim"] = out_type["shape"][axis_wo_batch]
    return Data(**out_type)


class LinearLayer(_ConcatInputLayer):
  layer_class = "linear"

  def __init__(self, activation, with_bias=True, grad_filter=None, **kwargs):
    """
    :param str|None activation: e.g. "relu", or None 
    :param bool with_bias: 
    :param float|None grad_filter: if grad norm is higher than this threshold (before activation), the grad is removed
    """
    super(LinearLayer, self).__init__(**kwargs)

    self.activation = activation
    self.with_bias = with_bias

    input_data = self.input_data
    n_in = input_data.dim
    n_out = self.output.dim
    assert n_in and n_out, "%r and %r" % (input_data, self.output)

    with var_creation_scope():
      W = self.add_param(tf.Variable(
        name="W",
        initial_value=tf.contrib.layers.xavier_initializer(seed=self.network.random.randint(2**31))(
          shape=(n_in, n_out))))

      if self.with_bias:
        b = self.add_param(tf.Variable(
          name="b",
          initial_value=tf.constant_initializer(value=0, dtype=tf.float32)(
            shape=(n_out,))))
      else:
        b = None

    with tf.name_scope("linear"):
      from TFUtil import dot
      x = input_data.placeholder
      ndim = x.get_shape().ndims

      if self.input_data.sparse:
        if x.dtype in [tf.uint8, tf.int8, tf.uint16, tf.int16]:
          x = tf.cast(x, tf.int32)
        x = tf.nn.embedding_lookup(W, x)
        ndim += 1
      else:
        x = dot(x, W)
      assert x.get_shape().ndims == ndim

      if self.with_bias:
        x = tf.add(x, b, name="add_bias")
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
  layer_class = "softmax"

  def __init__(self, activation="softmax", **kwargs):
    super(SoftmaxLayer, self).__init__(activation=activation, **kwargs)


class ConstantLayer(LayerBase):
  layer_class = "constant"

  def __init__(self, sources, value=0, dtype="float32", **kwargs):
    assert not sources
    super(ConstantLayer, self).__init__(**kwargs)
    # Add batch-dim to the constant.
    self.output.placeholder = tf.expand_dims(tf.constant(value, dtype=dtype), axis=0)

  @classmethod
  def get_out_data_from_opts(cls, name, dtype="float32", **kwargs):
    return Data(
      name="%s_const", shape=(), batch_dim_axis=0, time_dim_axis=None, dtype=dtype)


class GatingLayer(_ConcatInputLayer):
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
  def get_out_data_from_opts(cls, name, sources, n_out=None, **kwargs):
    input_data = get_concat_sources_data_template(sources)
    assert not input_data.sparse
    assert input_data.dim % 2 == 0
    dim = input_data.dim // 2
    if n_out:
      assert n_out == dim
    return Data(
      name="%s_output" % name,
      dtype=input_data.dtype,
      shape=input_data.shape[:-1] + (dim,),
      sparse=False,
      batch_dim_axis=input_data.batch_dim_axis,
      time_dim_axis=input_data.time_dim_axis)


class ConvLayer(_ConcatInputLayer):
  """
  A generic convolution layer which supports 1D, 2D and 3D convolution.
  Pooling can be done in the separate "pool" layer.
  """

  layer_class = "conv"
  recurrent = True  # we must not allow any shuffling in the time-dim or so

  def __init__(self, n_out, filter_size, padding, strides=1, dilation_rate=1,
               input_expand_dims=0, input_add_feature_dim=False, input_split_feature_dim=None,
               with_bias=False,
               activation=None,
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
    :param None|int input_split_feature_dim: if set, like input_add_feature_dim it will add a new feature dim
      which is of value input_split_feature_dim, and the original input feature dim
      will be divided by input_split_feature_dim, thus it must be a multiple of that value.
    :param bool with_bias: if True, will add a bias to the output features
    :param None|str activation: if set, will apply this function at the end
    """
    from TFUtil import check_input_dim, get_shape
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
    # We want to prepare the input data such that the batch-dim is the very first,
    # the feature-dim is the very last, and all in between are where we convolve over.
    # In the common terminology, this is the "NHWC" format, which is the default for TF convolution.
    x = self.input_data.get_placeholder_as_batch_major()
    x = check_input_dim(x, -1, self.input_data.dim)
    input_num_features = self.input_data.dim
    dyn_axes = self.input_data.get_dynamic_axes()  # conv-dims, or also called spatial dims
    static_axes = self.input_data.get_non_dynamic_axes()  # feature-dims
    assert dyn_axes + static_axes == list(range(self.input_data.ndim)), (
      "we expect the static dims at the end. input data is: %r" % self.input_data.get_description())
    if input_split_feature_dim:
      # Split the last two dimensions.
      assert self.input_data.dim % input_split_feature_dim == 0, "must be a multiple of the input feature dim"
      x = tf.reshape(
        x, get_shape(x)[:-1] + [self.input_data.dim // input_split_feature_dim, input_split_feature_dim])
      static_axes += [x.get_shape().ndims - 2]  # last without batch-dim
      input_num_features = input_split_feature_dim
    if input_add_feature_dim:
      # Add a dimension at the very end; any other static dims will be used as dynamic dims below.
      x = tf.expand_dims(x, axis=x.get_shape().ndims, name="input_use_feature_dim")
      static_axes += [x.get_shape().ndims - 2]  # last without batch-dim
      input_num_features = 1
    if len(static_axes) > 1:
      # Just treat them as dynamic axes, except the last.
      dyn_axes += static_axes[:-1]
      del static_axes[:-1]
    assert len(static_axes) == 1, "this should be our single input feature dim now. otherwise use input_add_feature_dim"
    while input_expand_dims:
      x = tf.expand_dims(x, axis=len(dyn_axes) + 1, name="input_expand_dims")  # axis including batch-dim
      dyn_axes += [len(dyn_axes)]
      static_axes = [axis + 1 for axis in static_axes]
      input_expand_dims -= 1
    assert dyn_axes == list(range(len(filter_size))), (
      "filter-size-dimension does not match the input data. " +
      "this is %i-D conv but number of spatial dims is %i in the input %s. " % (
        len(filter_size), len(dyn_axes), self.input_data.get_description()) +
      "consider using input_expand_dims or input_add_feature_dim.")
    filter_shape = list(filter_size) + [input_num_features, n_out]
    with var_creation_scope():
      filters = self.add_param(tf.Variable(
        name="W",
        initial_value=tf.contrib.layers.xavier_initializer(seed=self.network.random.randint(2**31))(
          shape=filter_shape)))
    y = tf.nn.convolution(x, filter=filters, padding=padding, strides=strides, dilation_rate=dilation_rate)
    # y shape is [batch] + dynamic_dims + [n_out].
    if with_bias:
      with var_creation_scope():
        b = self.add_param(tf.Variable(
          name="bias",
          initial_value=tf.constant_initializer(value=0, dtype=tf.float32)(
            shape=(n_out,))))
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
      i: (self.input_data.size_placeholder[i] if i in self.input_data.size_placeholder else tf.shape(y)[i + 1])
      for i in dyn_axes}
    if padding == "SAME":
      pass
    elif padding == "VALID":
      for i, s in list(self.output.size_placeholder.items()):
        self.output.size_placeholder[i] = s - filter_size[i] + 1
    else:
      assert False

  @classmethod
  def _get_out_type_from_opts(cls, n_out, filter_size, **kwargs):
    return {
      "dim": n_out,
      "shape": [None] * len(filter_size) + [n_out],
      "batch_dim_axis": 0,
      "sparse": False}

  @classmethod
  def get_out_data_from_opts(cls, **kwargs):
    out_type = cls._get_out_type_from_opts(**kwargs)
    return super(ConvLayer, cls).get_out_data_from_opts(out_type=out_type, **kwargs)


class PoolLayer(_ConcatInputLayer):
  """
  A generic N-D pooling layer.
  This would usually be done after a convolution for down-sampling.
  """

  layer_class = "pool"
  recurrent = True  # we should not shuffle in the time-dimension

  def __init__(self, mode, pool_size, padding="VALID", dilation_rate=1, strides=None, **kwargs):
    """
    :param str mode: "max" or "avg"
    :param tuple[int] pool_size: shape of the window of each reduce
    :param str padding: "valid" or "same"
    :param tuple[int]|int dilation_rate:
    :param tuple[int]|int|None strides: in contrast to tf.nn.pool, the default (if it is None) will be set to pool_size
    """
    assert "n_out" not in kwargs
    assert "out_type" not in kwargs
    from TFUtil import check_input_dim
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
    # We want to prepare the input data such that the batch-dim is the very first,
    # the feature-dim is the very last, and all in between are where we convolve over.
    # In the common terminology, this is the "NHWC" format, which is the default for TF convolution/pooling.
    x = self.input_data.get_placeholder_as_batch_major()
    x = check_input_dim(x, -1, self.input_data.dim)
    y = tf.nn.pool(
      x, window_shape=pool_size, pooling_type=mode, padding=padding,
      dilation_rate=dilation_rate, strides=strides)
    # y shape is [batch] + spatial_dims + [n_out].
    self.output.placeholder = y
    self.output.size_placeholder = {
      i: (self.input_data.size_placeholder[i] if i in self.input_data.size_placeholder else tf.shape(y)[i + 1])
      for i in range(len(pool_size))}
    if padding == "SAME":
      pass
    elif padding == "VALID":
      for i, s in list(self.output.size_placeholder.items()):
        self.output.size_placeholder[i] = s - pool_size[i] + 1
    else:
      assert False

  @classmethod
  def get_out_data_from_opts(cls, name, pool_size, sources, **kwargs):
    # y shape is [batch] + spatial_dims + [n_out].
    input_data = get_concat_sources_data_template(sources)
    return Data(
      name="%s_output" % name,
      shape=(None,) * len(pool_size) + (input_data.dim,),
      dim=input_data.dim,
      dtype=input_data.dtype,
      sparse=False,
      batch_dim_axis=0)


class ReduceLayer(_ConcatInputLayer):
  layer_class = "reduce"

  def __init__(self, mode, axis, keep_dims=False, enforce_batch_dim_axis=0, **kwargs):
    """
    :param str mode: "sum" or "max"
    :param int|list[int]|str axis: one axis or multiple axis to reduce.
      this is counted with batch-dim, which by default is axis 0 (see enforce_batch_dim_axis).
      it also accepts the special tokens "B"|"batch", "spatial", "spatial_except_time", or "F"|"feature"
    :param bool keep_dims: if dimensions should be kept (will be 1)
    :param int enforce_batch_dim_axis: will swap the batch-dim-axis of the input with the given axis.
      e.g. 0: will convert the input into batch-major format if not already like that.
    """
    from TFUtil import swapaxes
    assert "n_out" not in kwargs
    assert "out_type" not in kwargs
    mode = mode.lower()
    assert mode in ["max", "sum", "avg", "mean"]
    super(ReduceLayer, self).__init__(**kwargs)
    assert not self.input_data.sparse
    x = self.input_data.placeholder
    if self.input_data.batch_dim_axis != enforce_batch_dim_axis:
      x = swapaxes(x, self.input_data.batch_dim_axis, enforce_batch_dim_axis)
    axis = self.get_axes(axis, input_data=self.input_data)
    if mode == "max":
      f = tf.reduce_max
    elif mode == "sum":
      f = tf.reduce_sum
    elif mode in ["avg", "mean"]:
      f = tf.reduce_mean
    else:
      assert False
    y = f(x, axis=axis, keep_dims=keep_dims)
    y_dyn_sizes = self.input_data.size_placeholder.copy()
    if keep_dims:
      for i in axis:
        if i in y_dyn_sizes:
          y_dyn_sizes[i] = 1
    else:
      for i in reversed(sorted(axis)):
        if i in y_dyn_sizes:
          del y_dyn_sizes[i]
        y_dyn_sizes = {(j if (j < i) else (j - 1)): s
                       for (j, s) in list(y_dyn_sizes.items())}
    self.output.placeholder = y
    self.output.size_placeholder = y_dyn_sizes

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
  def get_out_data_from_opts(cls, name, sources, axis, keep_dims=False, enforce_batch_dim_axis=0, **kwargs):
    input_data = get_concat_sources_data_template(sources)
    axis = cls.get_axes(axis=axis, input_data=input_data)
    y_shape = list(input_data.batch_shape)
    if input_data.batch_dim_axis != enforce_batch_dim_axis:
      y_shape[input_data.batch_dim_axis], y_shape[enforce_batch_dim_axis] = \
        y_shape[enforce_batch_dim_axis], y_shape[input_data.batch_dim_axis]
    if keep_dims:
      for i in axis:
        y_shape[i] = 1
    else:
      for i in reversed(sorted(axis)):
        assert i != enforce_batch_dim_axis
        del y_shape[i]
    return Data(
      name="%s_output" % name,
      shape=tuple(y_shape[:enforce_batch_dim_axis] + y_shape[enforce_batch_dim_axis + 1:]),
      batch_dim_axis=enforce_batch_dim_axis,
      dtype=input_data.dtype,
      sparse=False)


class SqueezeLayer(_ConcatInputLayer):
  layer_class = "squeeze"

  def __init__(self, axis, enforce_batch_dim_axis=0, **kwargs):
    """
    :param int|list[int]|str axis: one axis or multiple axis to squeeze.
      this is counted with batch-dim, which by default is axis 0 (see enforce_batch_dim_axis).
      it also accepts the special tokens "B"|"batch", "spatial", "spatial_except_time", or "F"|"feature"
    """
    super(SqueezeLayer, self).__init__(**kwargs)
    axes = ReduceLayer.get_axes(axis, input_data=self.input_data)
    x = self.input_data.placeholder
    if self.input_data.batch_dim_axis != enforce_batch_dim_axis:
      x = swapaxes(x, self.input_data.batch_dim_axis, enforce_batch_dim_axis)
    for i in reversed(sorted(axes)):
      x = tf.squeeze(x, axis=i)
    self.output.placeholder = x

  @classmethod
  def get_out_data_from_opts(cls, **kwargs):
    return ReduceLayer.get_out_data_from_opts(keep_dims=False, **kwargs)


class PrefixInTimeLayer(CopyLayer):
  layer_class = "prefix_in_time"

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


class ResizeLayer(_ConcatInputLayer):
  layer_class = "resize"

  def __init__(self, factor, axis, kind="nn", **kwargs):
    """
    :param int factor:
    :param str|int axis: the axis to resize, counted with batch-dim. can also be "T" for time
    :param str kind: "linear", "nn"/"nearest_neighbor", "cubic"
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
    x = tf.reshape(x, [shape[0], shape[axis], 1] + [tf.reduce_prod([shape[i] for i in remaining_axes])])  # [batch,height,width,channels]
    new_size = shape[axis] * factor
    if kind == "linear":
      x = tf.image.resize_bilinear(x, size=(new_size, 1))
    elif kind == "cubic":
      x = tf.image.resize_bicubic(x, size=(new_size, 1))
    elif kind in ["nn", "nearest_neighbor"]:
      x = tf.image.resize_nearest_neighbor(x, size=(new_size, 1))
    else:
      raise Exception("invalid kind %r for resizing" % kind)
    x = tf.reshape(x, [shape[0], new_size, 1] + [shape[i] for i in remaining_axes])  # [batch,new_size,1] + remaining_axes
    x = tf.squeeze(x, axis=2)  # [batch,new_size] + remaining_axes
    if axis != 1:
      perm = [0] + remaining_axes
      perm.insert(axis, 1)
      x = tf.transpose(x, perm)
    self.output.placeholder = x

  @classmethod
  def get_out_data_from_opts(cls, factor, axis, sources, **kwargs):
    out = get_concat_sources_data_template(sources).copy_as_batch_major()
    axis = out.get_axis_from_description(axis)
    assert axis > 0, "batch-dim resize not supported"
    if out.shape[axis - 1] is not None:
      out_shape = list(out.shape)
      out_shape[axis - 1] *= factor
      out.shape = tuple(out_shape)
    return out


class CombineDimsLayer(_ConcatInputLayer):
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


class GetLastHiddenStateLayer(LayerBase):
  """
  Will combine (concat or add or so) all the last hidden states from all sources.
  """

  layer_class = "get_last_hidden_state"

  def __init__(self, n_out, combine="concat", **kwargs):
    """
    :param int n_out: dimension. output will be of shape (batch, n_out)
    :param str combine: "concat" or "add"
    """
    super(GetLastHiddenStateLayer, self).__init__(**kwargs)
    assert len(self.sources) > 0
    sources = [s.get_last_hidden_state() for s in self.sources]
    assert all([s is not None for s in sources])
    if len(sources) == 1:
      h = sources[0]
    else:
      if combine == "concat":
        h = tf.concat(sources, axis=1, name="concat_hidden_states")
      elif combine == "add":
        h = tf.add_n(sources, name="add_hidden_states")
      else:
        raise Exception("invalid hidden states combine mode %r" % combine)
    from TFUtil import check_input_ndim, check_input_dim
    h = check_input_ndim(h, 2)
    h = check_input_dim(h, 1, n_out)
    self.output.placeholder = h

  def get_last_hidden_state(self):
    return self.output.placeholder

  @classmethod
  def get_out_data_from_opts(cls, n_out, **kwargs):
    return super(GetLastHiddenStateLayer, cls).get_out_data_from_opts(
      out_type={"shape": (n_out,), "dim": n_out, "batch_dim_axis": 0, "time_dim_axis": None}, **kwargs)


class ChoiceLayer(LayerBase):
  """
  This layer represents a choice to be made in search during inference,
  such as choosing the top-k outputs from a log-softmax for beam search.
  During training, this layer can return the true label.
  This is supposed to be used inside the rec layer.
  This can be extended in various ways.
  
  Assume that we get input (batch,dim) from a log-softmax.
  Assume that each batch is already a choice via search.
  In search with a beam size of N, we would output
  sparse (batch=N,) and scores for each. 
  """
  layer_class = "choice"

  def __init__(self, beam_size, **kwargs):
    super(ChoiceLayer, self).__init__(**kwargs)
    # We assume log-softmax here, inside the rec layer.
    assert len(self.sources) == 1
    assert not self.sources[0].output.sparse
    assert self.sources[0].output.dim == self.output.dim
    assert self.sources[0].output.shape == (self.output.dim,)
    assert self.target
    self.choice_base_layer = self.network.get_search_choices(sources=self.sources)
    self.choice_scores = None
    self.choice_source_batches = None
    if self.network.search_flag:
      assert self.choice_base_layer  # not implemented yet...
      scores_base = self.choice_base_layer.rec_vars_outputs["choice_scores"]  # (batch,)
      scores_base = tf.expand_dims(scores_base, axis=-1)  # (batch, dim)
      scores_in = self.sources[0].output.placeholder  # (batch, dim)
      scores_in += scores_base  # (batch, dim)
      assert scores_in.get_shape().ndims == 2
      scores_in_shape = tf.shape(scores_in)
      scores_in_batch = scores_in_shape[0]
      scores_in_dim = scores_in_shape[1]
      scores_in_flat = tf.reshape(scores_in, [scores_in_batch * scores_in_dim])  # (batch * dim,)
      # We get scores/labels of shape (beam_size,) with indices in [0..batch*dim-1].
      scores, labels = tf.nn.top_k(scores_in_flat, k=beam_size)
      self.choice_source_batches = labels // scores_in_batch  # (beam_size,) -> scores_in batch idx
      labels = labels % scores_in_batch  # (beam_size,) -> dim idx
      self.choice_scores = scores  # (beam_size,) -> log score
      self.rec_vars_outputs["choice_scores"] = scores
      self.output = Data(
        name="%s_choice_output" % self.name,
        batch_dim_axis=0,
        shape=(),
        sparse=True,
        dim=self.output.dim,
        placeholder=labels)
    else:
      # Note: If you want to do forwarding, without having the reference,
      # that wont work. You must do search in that case.
      self.output = self._static_get_target_value(
        target=self.target, network=self.network,
        mark_data_key_as_used=True)

  @classmethod
  def get_out_data_from_opts(cls, target, network, **kwargs):
    return cls._static_get_target_value(
      target=target, network=network,
      mark_data_key_as_used=False)

  @classmethod
  def get_rec_initial_extra_outputs(cls, network, beam_size, **kwargs):
    if not network.search_flag:
      return {}
    return {"choice_scores": tf.zeros([beam_size])}


class RnnCellLayer(_ConcatInputLayer):
  """
  Wrapper around tf.contrib.rnn.RNNCell.
  This will operate a single step, i.e. there is no time dimension,
  i.e. we expect a (batch,n_in) input, and our output is (batch,n_out).
  """

  layer_class = "rnn_cell"

  def __init__(self, n_out, unit, initial_state=None, unit_opts=None, **kwargs):
    """
    :param int n_out: so far, only output shape (batch,n_out) supported
    :param str|tf.contrib.rnn.RNNCell unit: e.g. "BasicLSTM" or "LSTMBlock"
    :param str|float initial_state: see self.get_rec_initial_state()
    :param dict[str]|None unit_opts:
    """
    super(RnnCellLayer, self).__init__(**kwargs)
    self._initial_state = initial_state
    with tf.variable_scope(
          "rec",
          initializer=tf.contrib.layers.xavier_initializer(
            seed=self.network.random.randint(2**31))) as scope:
      assert isinstance(scope, tf.VariableScope)
      scope_name_prefix = scope.name + "/"  # e.g. "layer1/rec/"
      self.cell = self._get_cell(n_out=n_out, unit=unit, unit_opts=unit_opts)
      self.output.time_dim_axis = None
      self.output.batch_dim_axis = 0
      prev_state = self._rec_previous_layer.rec_vars_outputs["state"]
      self.output.placeholder, state = self.cell(self.input_data.placeholder, prev_state)
      self._hidden_state = state
      self.rec_vars_outputs["state"] = state
      params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name_prefix)
      assert params
      self.params.update({p.name[len(scope_name_prefix):-2]: p for p in params})

  @classmethod
  def _get_cell(cls, n_out, unit, unit_opts=None):
    """
    :param int n_out:
    :param str|tf.contrib.rnn.RNNCell unit:
    :param dict[str]|None unit_opts:
    :rtype: tf.contrib.rnn.RNNCell
    """
    import tensorflow.contrib.rnn as rnn_contrib
    if isinstance(unit, rnn_contrib.RNNCell):
      return unit
    rnn_cell_class = RecLayer.get_rnn_cell_class(unit)
    assert issubclass(rnn_cell_class, rnn_contrib.RNNCell)
    if unit_opts is None:
      unit_opts = {}
    assert isinstance(unit_opts, dict)
    # This should not have any side-effects, i.e. it should not add to the current computation graph,
    # it should also not create any vars yet, etc.
    cell = rnn_cell_class(n_out, **unit_opts)
    assert isinstance(cell, rnn_contrib.RNNCell)
    return cell

  @classmethod
  def get_out_data_from_opts(cls, n_out, name, **kwargs):
    return Data(
      name="%s_output" % name,
      shape=(n_out,), dim=n_out,
      batch_dim_axis=0,
      time_dim_axis=None,
      size_placeholder={})

  @classmethod
  def get_hidden_state_size(cls, n_out, unit, unit_opts=None, **kwargs):
    cell = cls._get_cell(unit=unit, unit_opts=unit_opts, n_out=n_out)
    import tensorflow.contrib.rnn as rnn_contrib
    assert isinstance(cell, rnn_contrib.RNNCell)
    return cell.state_size

  def get_hidden_state(self):
    return self._hidden_state

  def get_last_hidden_state(self):
    from tensorflow.python.util import nest
    if nest.is_sequence(self._hidden_state):
      return tf.concat(self._hidden_state, axis=1)
    return self._hidden_state

  @classmethod
  def _get_rec_initial_state(cls, batch_dim, name, initial_state=None, **kwargs):
    v = initial_state
    if isinstance(v, tf.Tensor):
      return v
    dim = cls.get_hidden_state_size(**kwargs)

    def make(d):
      assert isinstance(d, int)
      shape = [batch_dim, d]
      if v == "zeros" or not v:
        return tf.zeros(shape)
      elif v == "ones" or v == 1:
        return tf.ones(shape)
      else:
        raise Exception("invalid initial state type %r for sub-layer %r" % (v, name))

    if isinstance(dim, (tuple, list)):
      s = [make(d) for d in dim]
      # Make it the same type because nest.assert_same_structure() will complain otherwise.
      if isinstance(dim, tuple) and type(dim) is not tuple:  # assume namedtuple
        return type(dim)(*s)
      return type(dim)(s)
    return make(dim)

  @classmethod
  def get_rec_initial_extra_outputs(cls, **kwargs):
    return {"state": cls._get_rec_initial_state(**kwargs)}


class AttentionBaseLayer(_ConcatInputLayer):
  """
  This is the base class for attention.
  This layer would get constructed in the context of one single decoder step.
  We get the whole encoder output over all encoder frames (the base), e.g. (batch,enc_time,enc_dim),
  and some current decoder context, e.g. (batch,dec_att_dim),
  and we are supposed to return the attention output, e.g. (batch,att_dim).
  
  Some sources:
  * Bahdanau, Bengio, Montreal, Neural Machine Translation by Jointly Learning to Align and Translate, 2015, https://arxiv.org/abs/1409.0473
  * Luong, Stanford, Effective Approaches to Attention-based Neural Machine Translation, 2015, https://arxiv.org/abs/1508.04025
    -> dot, general, concat, location attention; comparison to Bahdanau
  * https://github.com/ufal/neuralmonkey/blob/master/neuralmonkey/decoders/decoder.py
  * https://google.github.io/seq2seq/
    https://github.com/google/seq2seq/blob/master/seq2seq/contrib/seq2seq/decoder.py
    https://github.com/google/seq2seq/blob/master/seq2seq/decoders/attention_decoder.py
  * https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/attention.py
  """

  def __init__(self, base, **kwargs):
    """
    :param LayerBase base: encoder output to attend on
    """
    super(AttentionBaseLayer, self).__init__(**kwargs)
    self.base = base

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    super(AttentionBaseLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["base"] = get_layer(d["base"])

  @classmethod
  def get_out_data_from_opts(cls, base, n_out=None, **kwargs):
    """
    :param LayerBase base:
    :rtype: Data
    """
    out = base.output.copy_template_excluding_time_dim()
    assert out.time_dim_axis is None
    if n_out:
      assert out.dim == n_out, (
        "The default attention selects some frame-weighted input of shape [batch, frame, dim=%i]," % out.dim +
        " thus resulting in [batch, dim=%i] but you specified n_out=%i." % (out.dim, n_out))
    return out


class GlobalAttentionContextBaseLayer(AttentionBaseLayer):
  def __init__(self, base_ctx, **kwargs):
    """
    :param LayerBase base_ctx: encoder output used to calculate the attention weights
    """
    super(GlobalAttentionContextBaseLayer, self).__init__(**kwargs)
    self.base_ctx = base_ctx

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    super(GlobalAttentionContextBaseLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["base_ctx"] = get_layer(d["base_ctx"])


class DotAttentionLayer(GlobalAttentionContextBaseLayer):
  """
  Classic global attention: Dot-product as similarity measure between base_ctx and source.
  """

  layer_class = "dot_attention"

  def __init__(self, **kwargs):
    super(DotAttentionLayer, self).__init__(**kwargs)
    # We expect input_data of shape (batch, inner),
    # base_ctx of shape (batch, base_time, inner) and base of shape (batch, base_time, n_out).
    assert self.input_data.batch_ndim == 2
    assert self.input_data.time_dim_axis is None
    assert self.base.output.batch_ndim == 3
    assert self.base_ctx.output.batch_ndim == 3
    assert self.input_data.dim == self.base_ctx.output.dim
    # And we want to do a dot product so that we get (batch, base_time).
    with tf.name_scope("att_energy"):
      # Get base of shape (batch, base_time, inner).
      base = self.base.output.get_placeholder_as_batch_major()  # (batch, base_time, n_out)
      base_ctx = self.base_ctx.output.get_placeholder_as_batch_major()  # (batch, base_time, inner)
      # Get source of shape (batch, inner, 1).
      source = tf.expand_dims(self.input_data.placeholder, axis=2)
      energy = tf.matmul(base_ctx, source)
      energy.set_shape(tf.TensorShape([None, None, 1]))  # (batch, base_time, 1)
      energy = tf.squeeze(energy, axis=2)  # (batch, base_time)
      base_weights = tf.nn.softmax(energy)  # (batch, base_time)
      base_weights_bc = tf.expand_dims(base_weights, axis=1)  # (batch, 1, base_time)
      out = tf.matmul(base_weights_bc, base)  # (batch, 1, n_out)
      out.set_shape(tf.TensorShape([None, 1, self.output.dim]))
      out = tf.squeeze(out, axis=1)  # (batch, n_out)
      self.output.placeholder = out
      self.output.size_placeholder = {}


class GaussWindowAttentionLayer(AttentionBaseLayer):
  """
  Interprets the incoming source as the location (float32, shape (batch,))
  and returns a gauss-window-weighting of the base around the location.
  The window size is fixed (TODO: but the variance can optionally be dynamic).
  """

  layer_class = "gauss_window_attention"

  def __init__(self, window_size, std=1., inner_size=None, inner_size_step=0.5, **kwargs):
    """
    :param int window_size: the window size where the Gaussian window will be applied on the base
    :param float std: standard deviation for Gauss
    :param int|None inner_size: if given, the output will have an additional dimension of this size,
      where t is shifted by +/- inner_size_step around.
      e.g. [t-1,t-0.5,t,t+0.5,t+1] would be the locations with inner_size=5 and inner_size_step=0.5. 
    :param float inner_size_step: see inner_size above
    """
    super(GaussWindowAttentionLayer, self).__init__(**kwargs)
    from TFUtil import expand_dims_unbroadcast

    # Code partly adapted from our Theano-based AttentionTimeGauss.
    # The beam is the window around the location center.

    with tf.name_scope("base"):
      base = self.base.output.get_placeholder_as_time_major()  # (base_time,batch,n_in)
    with tf.name_scope("base_seq_lens"):
      base_seq_lens = self.base.output.size_placeholder[0]  # (batch,)
      base_seq_lens_bc = tf.expand_dims(base_seq_lens, axis=0)  # (beam,batch)

    with tf.name_scope("std"):
      # Fixed std for now.
      # std = std_min + a[:, 1] * (std_max - std_min)  # (batch,)
      std = tf.expand_dims(tf.convert_to_tensor(std), axis=0)  # (batch,)

    with tf.name_scope("t"):
      if self.input_data.shape == ():
        t = self.input_data.get_placeholder_as_batch_major()  # (batch,)
      else:
        assert self.input_data.shape == (1,)
        t = tf.squeeze(self.input_data.get_placeholder_as_batch_major(), axis=1)  # (batch,)
      # Now calculate int32 indices for the window.
      t_round = tf.cast(tf.round(t), tf.int32)  # (batch,)
    with tf.name_scope("idxs"):
      start_idxs = t_round - window_size // 2  # (batch,), beams, centered around t_int
      idxs_0 = tf.expand_dims(tf.range(window_size), axis=1)  # (beam,batch). all on cpu, but static, no round trip
      idxs = idxs_0 + tf.expand_dims(start_idxs, axis=0)  # (beam,batch). centered around t_int
    with tf.name_scope("beam"):
      # Handle clipping for idxs.
      cidxs = tf.clip_by_value(idxs, 0, tf.shape(base)[0] - 1)
      cidxs = tf.where(tf.less(cidxs, base_seq_lens_bc), cidxs, tf.ones_like(cidxs) * base_seq_lens_bc - 1)
      #cidxs = tf.Print(cidxs, ["i=", self.network.layers[":i"].output.placeholder, "t=", t, "cidxs=", cidxs[window_size // 2:]])
      # We don't have multi_batch_beam for TF yet.
      # But tf.gather_nd or so might anyway be better to use here.
      # If that will not result in a sparse gradient in the while-loop,
      # some slicing with min(idxs)..max(idxs) might be anther option to at least reduce it a bit.
      # Note that gather_nd is broken up to TF 1.0 for this use case (see test_TFUtil.py),
      # so you need TF >=1.1 here.
      assert [int(i) for i in tf.__version__.split(".")[:2]] >= [1, 1]
      batches_idxs = tf.range(tf.shape(cidxs)[1], dtype=tf.int32, name="batches_idxs")  # (batch,)
      batches_idxs_bc = expand_dims_unbroadcast(batches_idxs, axis=0, dim=tf.shape(cidxs)[0], name="batches_idxs_bc")  # (beam,batch)
      idxs_exp = tf.stack([cidxs, batches_idxs_bc], axis=2, name="idxs_exp")  # (beam,batch,2), where the 2 stands for (base_time,batch)
      # Thus K == 2. gather_nd out will be idxs_exp.shape[:2] + params.shape[2:] = (beam,batch,n_in).
      gathered = tf.gather_nd(base, idxs_exp)  # (beam,batch,n_in)

    with tf.name_scope("gauss_window"):
      # Gauss window
      idxs_tr_bc = dimshuffle(idxs, (1, 0, 'x'))  # (batch,beam,inner_size)
      std_t_bc = dimshuffle(std, (0, 'x', 'x'))  # (batch,beam,inner_size)
      t_bc = dimshuffle(t, (0, 'x', 'x'))  # (batch,beam,inner_size)
      if inner_size:
        assert isinstance(inner_size, int)
        t_offs = tf.convert_to_tensor([(i * inner_size_step - inner_size / 2.0) for i in range(inner_size)])  # (inner_size,)
        t_offs_bc = dimshuffle(t_offs, ('x', 'x', 0))  # (batch,beam,inner_size)
        t_bc += t_offs_bc
      f_e = tf.exp(-((t_bc - tf.cast(idxs_tr_bc, tf.float32)) ** 2) / (2 * std_t_bc ** 2))  # (batch,beam,inner_size)
      from math import pi, sqrt
      norm = 1. / (std_t_bc * sqrt(2. * pi))  # (batch,beam,inner_size)
      w_t = f_e * norm  # (batch,beam,inner_size)

    with tf.name_scope("att"):
      gathered_tr = dimshuffle(gathered, (1, 2, 'x', 0))  # (batch,n_in,1,beam)
      w_t_bc = expand_dims_unbroadcast(w_t, axis=1, dim=self.base.output.dim)  # (batch,n_in,beam,inner_size)
      att = tf.matmul(gathered_tr, w_t_bc)  # (batch,n_in,1,inner_size)
      att = tf.squeeze(att, axis=2)  # (batch,n_in,inner_size)
      if not inner_size:
        att = tf.squeeze(att, axis=2)  # (batch,n_in)
      else:
        att = tf.transpose(att, (0, 2, 1))  # (batch,inner_size,n_in)

    self.output.placeholder = att
    self.output.size_placeholder = {}

  @classmethod
  def get_out_data_from_opts(cls, inner_size=None, **kwargs):
    out = super(GaussWindowAttentionLayer, cls).get_out_data_from_opts(**kwargs)
    if inner_size:
      assert isinstance(inner_size, int)
      out.shape = out.shape[:-1] + (inner_size,) + out.shape[-1:]
    return out


class RecLayer(_ConcatInputLayer):
  layer_class = "rec"
  recurrent = True
  _rnn_cells_dict = {}

  @classmethod
  def _create_rnn_cells_dict(cls):
    import tensorflow.contrib.rnn as rnn_contrib
    import TFNativeOp
    def maybe_add(key, v):
      if isinstance(v, type) and issubclass(v, (rnn_contrib.RNNCell, rnn_contrib.FusedRNNCell, TFNativeOp.RecSeqCellOp)):
        name = key
        if name.endswith("Cell"):
          name = name[:-len("Cell")]
        name = name.lower()
        assert cls._rnn_cells_dict.get(name) in [v, None]
        cls._rnn_cells_dict[name] = v
    for key, v in vars(rnn_contrib).items():
      maybe_add(key, v)
    for key, v in vars(TFNativeOp).items():
      maybe_add(key, v)

  @classmethod
  def get_rnn_cell_class(cls, name):
    """
    :param str name: cell name, minus the "Cell" at the end
    :rtype: () -> tensorflow.contrib.rnn.RNNCell
    """
    if not cls._rnn_cells_dict:
      cls._create_rnn_cells_dict()
    return cls._rnn_cells_dict[name.lower()]

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param TFNetwork.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    super(RecLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    initial_state = d.pop("initial_state", None)
    if initial_state:
      if not isinstance(initial_state, (list, tuple)):
        initial_state = [initial_state]
      assert len(initial_state) > 0
      sources = [get_layer(s).get_last_hidden_state() for s in initial_state]
      if len(sources) == 1:
        d["initial_state"] = sources[0]
      else:
        d["initial_state"] = tf.concat(sources, axis=1, name="concat_hidden_state")
    if isinstance(d.get("unit"), dict):
      def sub_get_layer(name):
        if name.startswith("base:"):
          return get_layer(name[len("base:"):])
      for sub in d["unit"].values():
        assert isinstance(sub, dict)
        if "initial_state" in sub:
          assert isinstance(sub["initial_state"], str)
          if sub["initial_state"].startswith("base:"):
            sub["initial_state"] = get_layer(sub["initial_state"][len("base:"):]).get_last_hidden_state()
        if "class" in sub:
          class_name = sub["class"]
          cl = get_layer_class(class_name)
          # Operate on a copy because we will transform the dict later.
          # We only need this to resolve any other layer dependencies in the main network.
          cl.transform_config_dict(sub.copy(), network=network, get_layer=sub_get_layer)

  def __init__(self,
               unit="lstm",
               direction=None, input_projection=True,
               initial_state=None,
               **kwargs):
    """
    :param str|dict[str,dict[str]] unit: the RNNCell/etc name, e.g. "nativelstm". see comment below.
      alternatively a whole subnetwork, which will be executed step by step,
      and which can include "prev" in addition to "from" to refer to previous steps.
    :param int|None direction: None|1 -> forward, -1 -> backward
    :param bool input_projection: True -> input is multiplied with matrix. False only works if same input dim
    :param tf.Tensor|None initial_state:
    :param dict[str]|bool output_feed_back: feedback mode opts, see OutputFeedback
    :param dict[str]|bool attention: attention opts, see Attention
    :param dict[str] kwargs: passed on to base class
    """
    super(RecLayer, self).__init__(**kwargs)
    import tensorflow.contrib.rnn as rnn_contrib
    import TFNativeOp
    if direction is not None:
      assert direction in [-1, 1]
    self._last_hidden_state = None
    self._direction = direction
    self._initial_state = initial_state
    self._input_projection = input_projection
    with tf.variable_scope(
          "rec",
          initializer=tf.contrib.layers.xavier_initializer(
            seed=self.network.random.randint(2**31))) as scope:
      assert isinstance(scope, tf.VariableScope)
      self._rec_scope = scope
      scope_name_prefix = scope.name + "/"  # e.g. "layer1/rec/"
      self.cell = self._get_cell(unit)
      if isinstance(self.cell, (rnn_contrib.RNNCell, rnn_contrib.FusedRNNCell)):
        y = self._get_output_cell(self.cell)
      elif isinstance(self.cell, TFNativeOp.RecSeqCellOp):
        y = self._get_output_native_rec_op(self.cell)
      elif isinstance(self.cell, self.SubnetworkCell):
        y = self._get_output_subnet_unit(self.cell)
      else:
        raise Exception("invalid type: %s" % type(self.cell))
      self.output.time_dim_axis = 0
      self.output.batch_dim_axis = 1
      self.output.placeholder = y
      params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name_prefix)
      self.params.update({p.name[len(scope_name_prefix):-2]: p for p in params})

  @classmethod
  def get_out_data_from_opts(cls, unit, **kwargs):
    n_out = kwargs.get("n_out", None)
    out_type = kwargs.get("out_type", None)
    target = kwargs.get("target", None)
    loss = kwargs.get("loss", None)
    if out_type or n_out or target or loss:
      if out_type:
        assert out_type.get("time_dim_axis", 0) == 0
        assert out_type.get("batch_dim_axis", 1) == 1
      out = super(RecLayer, cls).get_out_data_from_opts(**kwargs)
    else:
      out = None
    if isinstance(unit, dict):  # subnetwork
      if "n_out" in unit["output"] or "out_type" in unit["output"] or "target" in unit["output"] or "loss" in unit["output"]:
        sub_out_type = unit["output"].get("out_type")
        if sub_out_type and "shape" in sub_out_type:
          sub_out_type = sub_out_type.copy()
          sub_out_type["shape"] = (None,) + tuple(sub_out_type["shape"])  # add time-dim
        sub_opts = kwargs.copy()
        sub_opts.setdefault("n_out", unit["output"].get("n_out"))
        sub_opts.setdefault("out_type", sub_out_type)
        sub_opts.setdefault("target", unit["output"].get("target"))
        sub_opts.setdefault("loss", unit["output"].get("loss"))
        sub_out = cls._base_get_out_data_from_opts(**sub_opts)
        if out:
          assert sub_out.dim == out.dim
          assert sub_out.shape == out.shape
        else:
          out = sub_out
    assert out
    out.time_dim_axis = 0
    out.batch_dim_axis = 1
    return out

  def _get_input(self):
    """
    :return: (x, seq_len), where x is (time,batch,...,dim) and seq_len is (batch,)
    :rtype: (tf.Tensor, tf.Tensor)
    """
    assert self.input_data
    x = self.input_data.placeholder  # (batch,time,dim) or (time,batch,dim)
    if not self.input_data.is_time_major:
      assert self.input_data.batch_dim_axis == 0
      assert self.input_data.time_dim_axis == 1
      from TFUtil import swapaxes
      x = swapaxes(x, 0, 1)  # (time,batch,[dim])
    seq_len = self.input_data.size_placeholder[0]
    return x, seq_len

  def _get_cell(self, unit):
    import tensorflow.contrib.rnn as rnn_contrib
    import TFNativeOp
    if isinstance(unit, dict):
      return self.SubnetworkCell(parent=self, net_dict=unit)
    else:
      assert isinstance(unit, str)
      if unit.lower() in ["lstmp", "lstm"]:
        # Some possible LSTM implementations are (in all cases for both CPU and GPU):
        # * BasicLSTM (the cell), via official TF, pure TF implementation
        # * LSTMBlock (the cell), via tf.contrib.rnn.
        # * LSTMBlockFused, via tf.contrib.rnn. should be much faster than BasicLSTM
        # * NativeLSTM, our own native LSTM. should be faster than LSTMBlockFused
        # We default to the fastest one, i.e. NativeLSTM.
        # Note that they are currently not compatible to each other, i.e. the way the parameters are represented.
        unit = "nativelstm"  # TFNativeOp.NativeLstmCell
    rnn_cell_class = self.get_rnn_cell_class(unit)
    n_hidden = self.output.dim
    cell = rnn_cell_class(n_hidden)
    assert isinstance(
      cell, (rnn_contrib.RNNCell, rnn_contrib.FusedRNNCell, TFNativeOp.RecSeqCellOp))  # e.g. BasicLSTMCell
    return cell

  def _get_output_cell(self, cell):
    from tensorflow.python.ops import rnn
    import tensorflow.contrib.rnn as rnn_contrib
    assert self.input_data
    assert not self.input_data.sparse
    x, seq_len = self._get_input()
    if self._direction == -1:
      x = tf.reverse_sequence(x, seq_lengths=seq_len, batch_dim=1, seq_dim=0)
    if isinstance(cell, rnn_contrib.RNNCell):  # e.g. BasicLSTMCell
      # Will get (time,batch,ydim).
      y, final_state = rnn.dynamic_rnn(
        cell=cell, inputs=x, time_major=True, sequence_length=seq_len, dtype=tf.float32,
        initial_state=self._initial_state)
      self._last_hidden_state = final_state
    elif isinstance(cell, rnn_contrib.FusedRNNCell):  # e.g. LSTMBlockFusedCell
      # Will get (time,batch,ydim).
      y, final_state = cell(
        inputs=x, sequence_length=seq_len, dtype=tf.float32,
        initial_state=self._initial_state)
      self._last_hidden_state = final_state
    else:
      raise Exception("invalid type: %s" % type(cell))
    if self._direction == -1:
      y = tf.reverse_sequence(y, seq_lengths=seq_len, batch_dim=1, seq_dim=0)
    return y

  def _get_output_native_rec_op(self, cell):
    from TFUtil import dot, sequence_mask_time_major, directed
    assert self.input_data
    x, seq_len = self._get_input()
    if self._input_projection:
      W = tf.get_variable(name="W", shape=(self.input_data.dim, cell.n_input_dim), dtype=tf.float32)
      if self.input_data.sparse:
        x = tf.nn.embedding_lookup(W, x)
      else:
        x = dot(x, W)
    else:
      assert not self.input_data.sparse
      assert self.input_data.dim == cell.n_input_dim
    b = tf.get_variable(name="b", shape=(cell.n_input_dim,), dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    x += b
    index = sequence_mask_time_major(seq_len, maxlen=tf.shape(x)[0])
    y, final_state = cell(
      inputs=directed(x, self._direction), index=directed(index, self._direction),
      initial_state=self._initial_state)
    self._last_hidden_state = final_state
    y = directed(y, self._direction)
    return y

  class SubnetworkCell(object):
    def __init__(self, parent, net_dict):
      """
      :param RecLayer parent:
      :param dict[str] net_dict:
      """
      from copy import deepcopy
      self.parent = parent
      self.net_dict = deepcopy(net_dict)
      from TFNetwork import TFNetwork, ExternData
      self.net = TFNetwork(
        extern_data=ExternData(),
        train_flag=parent.network.train_flag,
        search_flag=parent.network.search_flag,
        parent=parent)
      if parent.input_data:
        self.net.extern_data.data["source"] = \
          parent.input_data.copy_template_excluding_time_dim()
      if parent.target:
        self.net.extern_data.data[parent.target] = \
          parent.network.extern_data.data[parent.target].copy_template_excluding_time_dim()
      self.layer_data_templates = {}  # type: dict[str,RecLayer.SubnetworkCell.TemplateLayer]
      self.prev_layers_needed = set()  # type: set[str]
      self._construct_template()
      self._initial_outputs = None  # type: dict[str,tf.Tensor]
      self._initial_extra_outputs = None  # type: dict[str,dict[str,tf.Tensor|tuple[tf.Tensor]]]

    class TemplateLayer(LayerBase):
      _sub_class_cache = {}

      @classmethod
      def _get_sub_class(cls, layer_class):
        if layer_class in cls._sub_class_cache:
          return cls._sub_class_cache[layer_class]
        class SubTemplateLayer(cls, layer_class):
          pass
        cls._sub_class_cache[layer_class] = SubTemplateLayer
        return SubTemplateLayer

      def __init__(self, network, name):
        """
        :param TFNetwork.TFNetwork network:
        :param str name:
        """
        # Init with some dummy.
        super(RecLayer.SubnetworkCell.TemplateLayer, self).__init__(
          out_type={"shape": ()}, name=name, network=network)
        self.output.size_placeholder = {}  # must be initialized
        self.layer_class = ":uninitialized-template"
        self._layer_class = None
        self._kwargs = None

      def init(self, output, layer_class, template_type="template", **kwargs):
        """
        :param Data output: 
        :param type[LayerBase]|LayerBase layer_class: 
        :param str template_type: 
        """
        self.__class__ = self._get_sub_class(layer_class)
        self.layer_class = ":%s:%s" % (template_type, layer_class.layer_class)
        self.output = output
        if not self.output.size_placeholder:
          self.output.size_placeholder = {}
        self._layer_class = layer_class
        self._kwargs = kwargs
        self._kwargs["output"] = output

      def copy(self, template_type):
        """
        :param str template_type: e.g. "prev"
        :return: new _TemplateLayer
        :rtype: RecLayer.SubnetworkCell.TemplateLayer
        """
        l = RecLayer.SubnetworkCell.TemplateLayer(network=self.network, name=self.name)
        l.init(layer_class=self._layer_class, template_type=template_type, **self._kwargs)
        return l

    class StepIndexLayer(LayerBase):
      layer_class = ":i"

      def __init__(self, i, **kwargs):
        super(RecLayer.SubnetworkCell.StepIndexLayer, self).__init__(
          output=Data(name="i", shape=(), dtype="int32", sparse=False, placeholder=tf.expand_dims(i, axis=0)),
          **kwargs)

    def _construct_template(self):
      """
      Without creating any computation graph, create TemplateLayer instances.
      """
      def add_templated_layer(name, layer_class, **layer_desc):
        """
        :param str name:
        :param type[LayerBase]|LayerBase layer_class:
        :param dict[str] layer_desc:
        :rtype: LayerBase
        """
        layer = self.layer_data_templates[name]
        layer_desc = layer_desc.copy()
        layer_desc["name"] = name
        layer_desc["network"] = self.net
        output = layer_class.get_out_data_from_opts(**layer_desc)
        layer.init(layer_class=layer_class, output=output, **layer_desc)
        return self.layer_data_templates[name]

      def get_templated_layer(name):
        """
        :param str name:
        :rtype: RecLayer.SubnetworkCell.TemplateLayer|LayerBase
        """
        if name.startswith("prev:"):
          name = name[len("prev:"):]
          self.prev_layers_needed.add(name)
        if name in self.layer_data_templates:
          return self.layer_data_templates[name]
        if name.startswith("base:"):
          return self.parent.network.layers[name[len("base:"):]]
        # Need to create layer instance here now to not run into recursive loops.
        # We will extend it later in add_templated_layer().
        self.layer_data_templates[name] = self.TemplateLayer(name=name, network=self.net)
        return self.net._construct_layer(
          self.net_dict, name, get_layer=get_templated_layer, add_layer=add_templated_layer)

      assert not self.layer_data_templates, "do not call this multiple times"
      get_templated_layer("output")
      assert "output" in self.layer_data_templates

    def _construct(self, prev_outputs, prev_extra, data=None, classes=None, i=None):
      """
      :param dict[str,tf.Tensor] prev_outputs: outputs of the layers from the previous step
      :param dict[str,dict[str,tf.Tensor]] prev_extra: extra output / hidden states of the previous step for layers
      :param tf.Tensor|None data: optional source data, shape e.g. (batch,dim)
      :param tf.Tensor|None classes: optional target classes, shape e.g. (batch,) if it is sparse
      :param tf.Tensor|None i: loop counter
      """
      if data is not None:
        self.net.extern_data.data["source"].placeholder = data
      if classes is not None:
        self.net.extern_data.data[self.parent.target].placeholder = classes

      prev_layers = {}  # type: dict[str,RecLayer.SubnetworkCell.TemplateLayer]
      for name in prev_outputs.keys():
        layer = prev_layers[name] = self.layer_data_templates[name].copy(template_type="prev")
        layer.output.placeholder = prev_outputs[name]
        layer.output.placeholder.set_shape(tf.TensorShape(layer.output.batch_shape))
        assert layer.output.placeholder.dtype is tf.as_dtype(layer.output.dtype)
        layer.output.size_placeholder = {}  # must be set
      for name in prev_extra.keys():
        if name not in prev_layers:
          layer = prev_layers[name] = self.layer_data_templates[name].copy(template_type="prev")
        else:
          layer = prev_layers[name]
        layer.rec_vars_outputs = prev_extra[name]

      from copy import deepcopy
      net_dict = deepcopy(self.net_dict)
      for name in net_dict.keys():
        if name in prev_layers:
          net_dict[name]["rec_previous_layer"] = prev_layers[name]

      def get_layer(name):
        if name.startswith("prev:"):
          return prev_layers[name[len("prev:"):]]
        if name.startswith("base:"):
          return self.parent.network.layers[name[len("base:"):]]
        return self.net._construct_layer(net_dict, name=name, get_layer=get_layer)

      assert not self.net.layers, "do not call this multiple times"
      if i is not None:
        self.net.layers[":i"] = self.StepIndexLayer(i=i, name=":i", network=self.net)
      get_layer("output")
      assert "output" in self.net.layers
      # Might not be resolved otherwise:
      for name in self.prev_layers_needed:
        get_layer(name)

    def _get_init_output(self, name, batch_dim):
      """
      :param str name: layer name
      :param tf.Tensor batch_dim:
      :rtype: tf.Tensor
      """
      cl = self.layer_data_templates[name]._layer_class
      return cl.get_rec_initial_output(batch_dim=batch_dim, **self.layer_data_templates[name]._kwargs)

    def _get_init_extra_outputs(self, name, batch_dim):
      """
      :param str name: layer name
      :param tf.Tensor batch_dim:
      :rtype: tf.Tensor|tuple[tf.Tensor]
      """
      cl = self.layer_data_templates[name]._layer_class
      d = cl.get_rec_initial_extra_outputs(batch_dim=batch_dim, **self.layer_data_templates[name]._kwargs)
      return d

    def get_next_loop_vars(self, loop_vars, data=None, classes=None, i=None):
      """
      :param (list[tf.Tensor],list[tf.Tensor]) loop_vars: loop_vars from the previous step
      :param tf.Tensor|None data: optional source data, shape e.g. (batch,dim)
      :param tf.Tensor|None classes: optional target classes, shape e.g. (batch,) if it is sparse
      :param tf.Tensor|None i: loop counter
      :return: next loop_vars
      :rtype: (list[tf.Tensor],list[tf.Tensor|tuple[tf.Tensor]])
      """
      from TFUtil import identity_op_nested
      from Util import sorted_values_from_dict, dict_zip
      prev_outputs_flat, prev_extra_flat = loop_vars
      assert len(prev_outputs_flat) == len(self.prev_layers_needed)
      prev_outputs = {k: v for (k, v) in zip(sorted(self.prev_layers_needed), prev_outputs_flat)}
      with tf.name_scope("prev_outputs"):
        prev_outputs = {k: tf.identity(v, name=k) for (k, v) in prev_outputs.items()}
      assert len(prev_extra_flat) == len(self._initial_extra_outputs)
      prev_extra = {
        k: dict_zip(sorted(self._initial_extra_outputs[k]), v)
        for (k, v) in zip(sorted(self._initial_extra_outputs), prev_extra_flat)}
      with tf.name_scope("prev_extra"):
        prev_extra = identity_op_nested(prev_extra)
      with reuse_name_scope(self.parent._rec_scope):
        self._construct(prev_outputs=prev_outputs, prev_extra=prev_extra, data=data, classes=classes, i=i)
      outputs_flat = [self.net.layers[k].output.placeholder for k in sorted(self.prev_layers_needed)]
      extra_flat = [
        sorted_values_from_dict(self.net.layers[k].rec_vars_outputs)
        for k in sorted(self.layer_data_templates)
        if self.net.layers[k].rec_vars_outputs]
      return outputs_flat, extra_flat

    def get_init_loop_vars(self, batch_dim):
      """
      :param tf.Tensor batch_dim: 
      :return: initial loop_vars. see self.get_next_loop_vars()
      :rtype: (list[tf.Tensor],list[tf.Tensor|tuple[tf.Tensor]])
      """
      self._initial_outputs = {k: self._get_init_output(k, batch_dim=batch_dim) for k in self.prev_layers_needed}
      self._initial_extra_outputs = {
        k: self._get_init_extra_outputs(k, batch_dim=batch_dim) for k in self.layer_data_templates.keys()}
      self._initial_extra_outputs = {k: v for (k, v) in self._initial_extra_outputs.items() if v}
      from Util import sorted_values_from_dict
      init_outputs_flat = sorted_values_from_dict(self._initial_outputs)
      init_extra_flat = [sorted_values_from_dict(v) for (k, v) in sorted(self._initial_extra_outputs.items())]
      return init_outputs_flat, init_extra_flat

  def _get_output_subnet_unit(self, cell):
    """
    :param RecLayer.SubnetworkCell cell:
    """
    with tf.name_scope("subnet_base"):
      if self.input_data:
        with tf.name_scope("x_tensor_array"):
          x, seq_len = self._get_input()  # x will be (time,batch,..,dim)
          x_shape = tf.shape(x)
          x_ta = tf.TensorArray(
            name="x_ta",
            dtype=self.input_data.dtype,
            element_shape=tf.TensorShape(self.input_data.copy_template_excluding_time_dim().batch_shape),
            size=x_shape[0],
            infer_shape=True)
          x_ta = x_ta.unstack(x)
        with tf.name_scope("batch_dim"):
          batch_dim = x_shape[1]
      else:
        x_ta = None
        seq_len = self.output.size_placeholder[0]  # see BaseLayer.__init__()
        data = self.network.extern_data.get_default_input_data()
        with tf.name_scope("batch_dim"):
          batch_dim = tf.shape(data.placeholder)[data.batch_dim_axis]
      max_out_len = tf.reduce_max(seq_len)

      # TODO: Better check for train_flag.
      # Maybe more generic via sampling options later.
      y_ta = None
      if self.target and self.network.train_flag is not False:
        y_data = self.network.get_extern_data(self.target, mark_data_key_as_used=True)
        y = y_data.get_placeholder_as_time_major()
        y_max_len = tf.shape(y)[0]
        with tf.control_dependencies([tf.assert_equal(max_out_len, y_max_len,
            ["RecLayer %r with sources %r." % (self.name, self.sources),
             " The length of the sources (", max_out_len,
             ") differ from the length of the target (", y_max_len, ")."])]):
          y_max_len = tf.identity(y_max_len)
        y_ta = tf.TensorArray(
          name="y_ta",
          dtype=y_data.dtype,
          element_shape=tf.TensorShape(y_data.copy_template_excluding_time_dim().batch_shape),
          size=y_max_len,
          infer_shape=True)
        y_ta = y_ta.unstack(y)

      # Note: tf.while_loop() will not give us all intermediate outputs, but we want them.
      # tf.scan() would do that but tf.scan() will loop over some input sequence -
      # however, that would not work because the input sequence is not fixed initially.
      # So, similar as tf.scan() does it, we collect all intermediate values.

      # In the while-loop, what we need to output is:
      # * next step counter (i)
      # * all outputs from layers which are in self.prev_layers_needed
      # * all hidden states from RnnCellLayer
      # * accumulated TensorArray of outputs from the output-layer for each step
      # For each of this, we need a sensible init, which we are supposed to return here.

      init_net_vars = cell.get_init_loop_vars(batch_dim=batch_dim)
      init_i = tf.constant(0)
      min_out_len = max_out_len

      from collections import namedtuple
      OutputToAccumulate = namedtuple("OutputToAccumulate", ["name", "dtype", "element_shape", "get"])
      outputs_to_accumulate = [
        OutputToAccumulate(
          name="output",
          dtype=cell.layer_data_templates["output"].output.dtype,
          element_shape=cell.layer_data_templates["output"].output.batch_shape,
          get=lambda: cell.net.layers["output"].output.placeholder)]

      if self.network.search_flag:
        for layer in cell.layer_data_templates.values():
          if isinstance(layer, ChoiceLayer):
            outputs_to_accumulate += [
              OutputToAccumulate(
                name="choice_%s" % layer.name,
                dtype=tf.int32,
                element_shape=(layer._kwargs["beam_size"],),
                get=(lambda name:
                  lambda: cell.net.layers[name].choice_source_batches)(layer.name))]

      # Create a tensor array to store the intermediate values for each step i, e.g. of shape (batch, dim).
      init_acc_tas = [
        tf.TensorArray(
          name="acc_ta_%s" % out.name,
          dtype=out.dtype,
          element_shape=tf.TensorShape(out.element_shape),
          size=min_out_len,
          dynamic_size=True,  # we will automatically grow it when needed
          infer_shape=True)
        for out in outputs_to_accumulate]

    def body(i, net_vars, acc_tas):
      """
      The loop body of scan.

      :param tf.Tensor i: loop counter, scalar
      :param net_vars: the accumulator values
      :param list[tf.TensorArray] acc_tas: the output accumulator TensorArray
      :return: [i + 1, a_flat, tas]: the updated counter + new accumulator values + updated TensorArrays
      :rtype: (tf.Tensor, object, list[tf.TensorArray])

      Raises:
        TypeError: if initializer and fn() output structure do not match
        ValueType: if initializer and fn() output lengths do not match
      """
      # The inner scope name is a bit screwed up and this is nicer anyway.
      with reuse_name_scope(self._rec_scope.name + "/while_loop_body", absolute=True):
        net_vars = cell.get_next_loop_vars(
          net_vars,
          data=x_ta.read(i) if x_ta else None,
          classes=y_ta.read(i) if y_ta else None,
          i=i)
        assert len(acc_tas) == len(outputs_to_accumulate)
        acc_tas = [
          acc_ta.write(i, out.get())
          for (acc_ta, out) in zip(acc_tas, outputs_to_accumulate)]
        return i + 1, net_vars, acc_tas

    def cond(i, net_vars, acc_ta):
      return tf.less(i, max_out_len)

    _, _, final_acc_tas = tf.while_loop(cond=cond, body=body, loop_vars=(init_i, init_net_vars, init_acc_tas))
    assert len(final_acc_tas) == len(outputs_to_accumulate)
    final_outputs = {
      out.name: final_acc_ta.stack()  # e.g. (time, batch, dim)
      for (final_acc_ta, out) in zip(final_acc_tas, outputs_to_accumulate)}
    return final_outputs["output"]

  def get_last_hidden_state(self):
    assert self._last_hidden_state is not None, (
      "last-hidden-state not implemented/supported for this layer-type. try another unit. see the code.")
    return self._last_hidden_state


class FsaLayer(LayerBase):
  layer_class = "fsa"

  def __init__(self, **kwargs):
    """
    """
    super(FsaLayer, self).__init__(**kwargs)
    # TODO...


class CombineLayer(LayerBase):
  layer_class = "combine"

  def _check_same_dense_dim(self, sources):
    """
    :param list[LayerBase] sources:
    """
    assert not self.output.sparse
    for source in sources:
      assert not source.output.sparse
      assert source.output.dim == self.output.dim

  # Requires the same input shape and yield the same output shape.
  def _op_kind_add(self, sources):
    """
    :param list[LayerBase] sources:
    :rtype: tf.Tensor
    """
    self._check_same_dense_dim(sources)
    from TFUtil import swapaxes
    x = sources[0].output.placeholder
    batch_axis = sources[0].output.batch_dim_axis
    for source in sources[1:]:
      x2 = source.output.placeholder
      if source.output.batch_dim_axis != batch_axis:
        x2 = swapaxes(x2, batch_axis, source.output.batch_dim_axis)
      x += x2
    return x

  # Requires the same input shape and yield the same output shape.
  def _op_kind_average(self, sources):
    """
    :param list[LayerBase] sources:
    :rtype: tf.Tensor
    """
    x = self._op_kind_add(sources)
    x /= len(sources)
    return x

  def __init__(self, kind, sources, activation=None, with_bias=False, **kwargs):
    """
    :param str kind: e.g. "average"
    :param list[LayerBase] sources:
    :param str|None activation: if provided, activation function to apply, e.g. "tanh" or "relu"
    :param bool with_bias: if given , will add a bias
    """
    assert sources
    super(CombineLayer, self).__init__(sources=sources, **kwargs)
    op = getattr(self, "_op_kind_%s" % kind)
    x = op(sources)
    if with_bias:
      with var_creation_scope():
        b = self.add_param(tf.Variable(
          name="b",
          initial_value=tf.constant_initializer(value=0, dtype=tf.float32)(
            shape=(self.output.dim,))))
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
  def get_out_data_from_opts(cls, n_out=None, out_type=None, sources=(), **kwargs):
    if not n_out and not out_type:
      out_type = sources[0].output.get_kwargs()
      out_type["name"] = "%s_output" % kwargs["name"]
    return super(CombineLayer, cls).get_out_data_from_opts(n_out=n_out, out_type=out_type, sources=sources, **kwargs)


class SubnetworkLayer(LayerBase):
  """
  You can define a whole subnetwork as a single layer by this class.
  """

  layer_class = "subnetwork"
  recurrent = True  # we don't know. depends on the subnetwork.

  def __init__(self, subnetwork, concat_sources=True, **kwargs):
    """
    :param dict[str,dict] network: subnetwork as dict (JSON content). must have an "output" layer
    :param bool concat_sources: if we concatenate all sources into one, like it is standard for most other layers
    """
    super(SubnetworkLayer, self).__init__(**kwargs)
    from TFNetwork import TFNetwork, ExternData
    sub_extern_data = ExternData()
    if concat_sources:
      sub_extern_data.data[sub_extern_data.default_input] = \
        concat_sources_with_opt_dropout(self.sources, dropout=kwargs.get("dropout", 0))
    else:
      assert not kwargs.get("dropout", 0), "not supported without concat_sources"
      for source in self.sources:
        assert isinstance(source, LayerBase)
        sub_extern_data.data[source.name] = source.output
    net = TFNetwork(
      rnd_seed=self.network.random.randint(2**31),
      train_flag=self.network.train_flag,
      extern_data=sub_extern_data,
      parent=self)
    net.construct_from_dict(subnetwork)
    self.subnetwork = net
    self.output = net.get_default_output_layer().output
    for layer in net.layers.values():
      assert layer.trainable == self.trainable, "partly trainable subnetworks not yet supported"
      self.params.update({"%s/%s" % (layer.name, k): v for (k, v) in layer.params.items()})

  @classmethod
  def get_out_data_from_opts(cls, subnetwork, n_out=None, out_type=None, **kwargs):
    """
    :param dict[str,dict[str]] subnetwork:
    :param int|None n_out:
    :param dict[str]|None out_type:
    :rtype: Data
    """
    if n_out or out_type:
      return super(SubnetworkLayer, cls).get_out_data_from_opts(n_out=n_out, out_type=out_type, **kwargs)
    layer_desc = subnetwork["output"].copy()
    class_name = layer_desc.pop("class")
    layer_class = get_layer_class(class_name)
    def _get_layer(name):
      raise Exception("not available at this point; provide n_out or out_type explicitly.")
    layer_desc["from"] = []  # that wont work here
    layer_class.transform_config_dict(layer_desc, get_layer=_get_layer, network=kwargs["network"])
    layer_desc["name"] = "output"
    layer_desc["network"] = None
    # Note: This can likely fail because we don't provide all the right args.
    # In that case, you must provide n_out or out_type explicitly.
    return layer_class.get_out_data_from_opts(**layer_desc)

  def get_constraints_value(self):
    self.subnetwork.maybe_construct_objective()
    v = self.subnetwork.total_constraints
    if v is 0:
      return None
    return v

  def get_loss_value(self):
    self.subnetwork.maybe_construct_objective()
    v = self.subnetwork.total_loss
    if v is 0:
      return None
    return v

  def get_error_value(self):
    self.subnetwork.maybe_construct_objective()
    errors = self.subnetwork.get_all_errors()
    if not errors:
      return None
    if len(errors) == 1:
      return list(errors.values())[0]
    name = self.subnetwork.get_default_output_layer_name()
    if name in errors:
      return errors[name]
    return sorted(errors.items())[0][1]  # first alphabetically

  def get_last_hidden_state(self):
    h = self.subnetwork.get_default_output_layer().get_last_hidden_state()
    if h is not None:
      return h
    return super(SubnetworkLayer, self).get_last_hidden_state()


class FramewiseStatisticsLayer(LayerBase):
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
    output_before_softmax_flat = flatten_with_seq_len_mask(source.output_before_activation.x, output_seq_lens, time_major=output.is_time_major)
    target_seq_lens = target.size_placeholder[0]
    target_flat = flatten_with_seq_len_mask(target.placeholder, target_seq_lens, time_major=target.is_time_major)
    target_flat.set_shape(tf.TensorShape([tf.Dimension(None)]))
    loss_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_before_softmax_flat, labels=target_flat)
    flat_last_dim = output_before_softmax_flat.get_shape().ndims - 1
    assert flat_last_dim == 1
    output_flat = flatten_with_seq_len_mask(output.placeholder, output_seq_lens, time_major=output.is_time_major)
    output_flat_argmax = tf.cast(tf.arg_max(output_before_softmax_flat, dimension=flat_last_dim), "int32")
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

    with var_creation_scope():
      accumulated_seq_len = tf.Variable(initial_value=0, dtype=tf.int64, trainable=False, name="accumulated_seq_len")
      accumulated_seq_len_sil = tf.Variable(initial_value=0, dtype=tf.int64, trainable=False, name="accumulated_seq_len_sil")
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
        with var_creation_scope():
          acc_v = tf.Variable(initial_value=numpy.zeros(acc_shape, dtype=acc_dtype), dtype=acc_dtype, trainable=False, name="accumulated_%s" % k)
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
    # n_out=1 is a workaround for now. Our output should not be used. We have none.
    return super(FramewiseStatisticsLayer, cls).get_out_data_from_opts(n_out=1, **kwargs)


class Loss(object):
  class_name = None
  recurrent = False  # if this is a frame-wise criteria, this will be False

  def __init__(self):
    # All are initialized in self.init().
    self.output = None  # type: Data
    self.time_major = None  # type: bool|None
    self.output_with_activation = None  # type: OutputWithActivation
    self.output_seq_lens = None  # type: tf.Tensor
    self.target = None  # type: Data
    self.target_seq_lens = None  # type: tf.Tensor
    self.output_flat = None  # type: tf.Tensor
    self.output_before_softmax_flat = None  # type: tf.Tensor
    self.target_flat = None  # type: tf.Tensor
    # Maybe make configurable. For now, same as in our Theano behavior.
    self.reduce_func = tf.reduce_sum  # or tf.reduce_mean
    self.loss_norm_factor = None  # type: tf.Tensor

  def init(self, output, output_with_activation=None, target=None):
    """
    :param Data output: generated output
    :param OutputWithActivation|None output_with_activation:
    :param Data target: reference target from dataset
    """
    from TFUtil import flatten_with_seq_len_mask
    with tf.name_scope("loss_init"):
      self.output = output
      self.output_with_activation = output_with_activation
      self.output_seq_lens = output.size_placeholder[0]
      self.target = target
      self.target_seq_lens = target.size_placeholder[0]
      # Flat variants are with batch,time collapsed into one, masked via seq_lens.
      self.output_flat = None
      self.output_before_softmax_flat = None
      if output_with_activation:
        assert output_with_activation.y is output.placeholder
      if self.output.have_tim_axis():
        self.loss_norm_factor = 1.0 / tf.cast(tf.reduce_sum(self.target_seq_lens), tf.float32)
        time_and_batch_dims = (self.output.time_dim_axis, self.output.batch_dim_axis)
        assert time_and_batch_dims in [(0, 1), (1, 0)], "output time-batch-dim unexpected: %s" % self.output
        if output_with_activation and output_with_activation.act_func is tf.nn.softmax:
          self.output_before_softmax_flat = flatten_with_seq_len_mask(output_with_activation.x, self.output_seq_lens, time_major=output.is_time_major)
        else:
          self.output_flat = flatten_with_seq_len_mask(output.placeholder, self.output_seq_lens, time_major=output.is_time_major)
          self.output_flat.set_shape(tf.TensorShape(output.shape))
      else:
        self.loss_norm_factor = 1.0
      self.target_flat = flatten_with_seq_len_mask(target.placeholder, self.target_seq_lens, time_major=target.is_time_major)
      self._check_init()

  def _check_init(self):
    """
    Does some checks on self.target and self.output, e.g. if the dense shapes matches.
    You can overwrite this if those checks don't make sense for your derived loss class.
    """
    assert self.target.ndim_dense == self.output.ndim_dense, (
      "Number of dimensions missmatch. Target: %s, output: %s" % (self.target, self.output))
    expected_output_dim = self.get_auto_output_layer_dim(self.target.dim)
    assert expected_output_dim == self.output.dim, (
      "Expected output dim is %i but the output has dim %i. " % (expected_output_dim, self.output.dim) +
      "Target: %s, output: %s" % (self.target, self.output))

  def get_error(self):
    """
    :return: frame error rate as a scalar value
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
        target_label = tf.cast(tf.arg_max(target_flat, dimension=last_dim), tf.int32)
      output_label = tf.cast(tf.arg_max(output_flat, dimension=last_dim), target_label.dtype)
      not_equal = tf.not_equal(output_label, target_label)
      return self.reduce_func(tf.cast(not_equal, tf.float32))

  def get_value(self):
    """
    :return: loss as a scalar value
    :rtype: tf.Tensor
    """
    raise NotImplementedError

  def get_normalization_factor(self):
    """
    :return: factor as a float scalar, usually 1.0 / num_frames. see self.reduce_func.
    :rtype: tf.Tensor
    """
    return self.loss_norm_factor

  def get_auto_output_layer_dim(self, target_dim):
    """
    :param int target_dim:
    :return: normally just the same as target_dim. e.g. for CTC, we would add 1 for the blank label
    :rtype: int
    """
    return target_dim


class CrossEntropyLoss(Loss):
  class_name = "ce"

  def get_value(self):
    with tf.name_scope("loss_ce"):
      assert self.target.ndim_dense == self.output.ndim_dense
      if self.target.sparse:
        if self.output_before_softmax_flat is not None:
          out = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.output_before_softmax_flat, labels=self.target_flat)
          return self.reduce_func(out)
        else:
          target_flat_exp = tf.stack(
            [tf.range(tf.shape(self.target_flat)[0], dtype=tf.int32),
             tf.cast(self.target_flat, tf.int32)], axis=1)  # (time,2)
          out = tf.log(tf.gather_nd(self.output_flat, target_flat_exp))
          return -self.reduce_func(out)
      else:  # not sparse
        if self.output_before_softmax_flat is not None:
          out = tf.nn.softmax_cross_entropy_with_logits(self.output_before_softmax_flat, self.target_flat)
          return self.reduce_func(out)
        else:
          out = self.target_flat * tf.log(self.output_flat)
          return -self.reduce_func(out)


class GenericCELoss(Loss):
  class_name = "generic_ce"

  def __init__(self, **kwargs):
    super(GenericCELoss, self).__init__(**kwargs)

    def loss(z, y, grad_f, target):
      nlog_scores = -tf.log(tf.clip_by_value(y, 1.e-20, 1.e20))  # (time,dim)
      # target is shape (time,) -> index.
      target_exp = tf.stack([tf.range(tf.shape(target)[0], dtype=tf.int32), target], axis=1)  # (time,2)
      # Thus K == 2. gather_nd out will be (target_exp.shape[0],) = (time,).
      gathered = tf.gather_nd(nlog_scores, target_exp)   # (time,)
      return self.reduce_func(gathered)

    def loss_grad(op, grad):
      """
      :param tf.Operation op:
      :param tf.Tensor grad: grad for loss
      :return: grad for op.inputs
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
  class_name = "ctc"
  recurrent = True

  def __init__(self, target_collapse_repeated=False, auto_clip_target_len=False):
    """
    :param bool target_collapse_repeated: like preprocess_collapse_repeated option for CTC. used for sparse_labels().
    :param bool auto_clip_target_len: see self._get_target_sparse_labels().
    """
    super(CtcLoss, self).__init__()
    self.target_collapse_repeated = target_collapse_repeated
    self.auto_clip_target_len = auto_clip_target_len
    self._target_sparse_labels = None

  def init(self, **kwargs):
    self._target_sparse_labels = None
    super(CtcLoss, self).init(**kwargs)

  def _get_target_sparse_labels(self):
    if self._target_sparse_labels is not None:
      return self._target_sparse_labels
    from TFUtil import sparse_labels
    target_seq_lens = self.target_seq_lens
    if self.auto_clip_target_len:
      # Not more than output_seq_lens, otherwise we can get an exception by the CTC algorithm
      # "Not enough time for target transition sequence".
      # One less to allow for at least one blank somewhere.
      target_seq_lens = tf.minimum(target_seq_lens, tf.maximum(self.output_seq_lens - 1, 0))
    labels = sparse_labels(self.target.placeholder, target_seq_lens,
                           collapse_repeated=self.target_collapse_repeated)
    self._target_sparse_labels = labels
    return labels

  def get_value(self):
    if not self.target.sparse:
      raise Exception("CTC target expected to be sparse (symbols)")
    with tf.name_scope("loss_ctc"):
      logits = self.output_with_activation
      if self.output_with_activation:
        logits = self.output_with_activation.get_logits()
      if logits is None:
        logits = tf.log(self.output.placeholder)
      assert logits.get_shape().ndims == 3  # (B,T,N) or (T,B,N)
      assert logits.get_shape().dims[2].value == self.target.dim + 1  # one more for blank
      seq_lens = self.output_seq_lens
      labels = self._get_target_sparse_labels()
      loss = tf.nn.ctc_loss(inputs=logits, labels=labels, sequence_length=seq_lens, time_major=self.output.is_time_major)
      return self.reduce_func(loss)

  def get_error(self):
    if not self.target.sparse:
      raise Exception("CTC target expected to be sparse (symbols)")
    with tf.name_scope("loss_ctc_error"):
      logits = None
      if self.output_with_activation:
        logits = self.output_with_activation.get_logits()
      if logits is None:
        logits = tf.log(self.output.placeholder)
      if not self.output.is_time_major:
        logits = tf.transpose(logits, [1, 0, 2])  # (B,T,N) => (T,B,N)
      seq_lens = self.output_seq_lens
      decoded, _ = tf.nn.ctc_greedy_decoder(inputs=logits, sequence_length=seq_lens)
      labels = self._get_target_sparse_labels()
      error = tf.edit_distance(hypothesis=tf.cast(decoded[0], labels.dtype), truth=labels, normalize=False)
      return self.reduce_func(error)

  def get_auto_output_layer_dim(self, target_dim):
    return target_dim + 1  # one added for blank


_LossClassDict = {}  # type: dict[str,type(Loss)]

def get_loss_class(loss):
  """
  :param str loss: loss type such as "ce"
  :rtype: () -> Loss
  """
  if not _LossClassDict:
    for v in globals().values():
      if isinstance(v, type) and issubclass(v, Loss) and v.class_name:
        assert v.class_name not in _LossClassDict
        _LossClassDict[v.class_name] = v
  return _LossClassDict[loss]


_LayerClassDict = {}  # type: dict[str,type(LayerBase)]

def _init_layer_class_dict():
  for v in globals().values():
    if isinstance(v, type) and issubclass(v, LayerBase) and v.layer_class:
      assert v.layer_class not in _LayerClassDict
      _LayerClassDict[v.layer_class] = v
  for alias, v in {"forward": LinearLayer}.items():
    assert alias not in _LayerClassDict
    _LayerClassDict[alias] = v


def get_layer_class(name):
  """
  :param str name: matches layer_class
  :rtype: (() -> LayerBase) | LayerBase
  """
  if not _LayerClassDict:
    _init_layer_class_dict()
  if name not in _LayerClassDict:
    raise Exception("unknown layer class %r" % name)
  return _LayerClassDict[name]

