
from __future__ import print_function

import tensorflow as tf
import contextlib
import TFUtil
from Util import unicode
from TFUtil import Data, OutputWithActivation, CustomUpdate, dimshuffle, swapaxes
from Log import log


class LayerBase(object):
  """
  This is the base class for all layers.
  Every layer by default has a list of source layers `sources` and defines `self.output` which is of type Data.
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

  layer_class = None  # type: str|None  # for get_layer_class()
  recurrent = False  # if the order in the time-dimension is relevant

  def __init__(self, name, network, output=None, n_out=None, out_type=None, sources=(),
               target=None, loss=None, loss_scale=1.0, size_target=None,
               reuse_params=None,
               L2=None, is_output_layer=None,
               copy_output_loss_from_source_idx=None,
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
    :param float loss_scale: scale factor for loss (1.0 by default)
    :param LayerBase|None reuse_params: if given, will reuse the params from this layer. see self.var_creation_scope()
    :param float|None L2: for constraints
    :param bool|None is_output_layer:
    :param int|None copy_output_loss_from_source_idx: if set, will copy output_loss from this source
    :param bool|dict batch_norm: see self.batch_norm()
    :param str|float initial_output: used for recurrent layer, see self.get_rec_initial_output()
    :param LayerBase|None rec_previous_layer: via the recurrent layer, layer (template) which represents the past of us
    :param bool trainable: whether the parameters of this layer will be trained
    """
    self.name = name
    self.network = network
    self.target = target
    self.loss = loss
    if self.loss and self.loss.recurrent:
      self.recurrent = True
    self.loss_scale = loss_scale
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
    self.output_loss = None  # type: None|tf.Tensor
    if copy_output_loss_from_source_idx is not None:
      self.output_loss = sources[copy_output_loss_from_source_idx].output_loss
    self.rec_vars_outputs = {}  # type: dict[str,tf.Tensor]
    self.search_choices = None  # type: SearchChoices
    self._initial_output = initial_output
    self._rec_previous_layer = rec_previous_layer
    self.sources = sources
    self.params = {}  # type: dict[str,tf.Variable]
    self.saveable_param_replace = {}  # see get_saveable_params_dict()
    " :type: dict[tf.Variable,tensorflow.python.training.saver.BaseSaverBuilder.SaveableObject] "
    self.reuse_params = reuse_params
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
    if n_out is None and target:
      n_out = cls._static_get_target_value(target=target, network=network, mark_data_key_as_used=False).dim
    if out_type is None:
      assert n_out
      out_type = {"dim": n_out}
    out_type = out_type.copy()
    out_type.setdefault("name", "%s_output" % name)
    if sources and not sources[0].output.sparse and not out_type.get("sparse", False):
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
    beam_size = None
    for src in sources:
      beam_size = beam_size or src.output.beam_size
    out_type.setdefault("beam_size", beam_size)
    output = Data(**out_type)
    cls._post_init_output(
      output=output, network=network, target=target, size_target=size_target, sources=sources, **kwargs)
    return output

  @classmethod
  def _post_init_output(cls, output, network, target=None, size_target=None, sources=(), **kwargs):
    """
    :param Data output:
    :param TFNetwork.TFNetwork network:
    :param str|None target:
    :param str|None size_target:
    :param list[LayerBase] sources:
    """
    # You are supposed to set self.output.placeholder to the value which you want to return by the layer.
    # Normally you are also supposed to set self.output.size_placeholder explicitly, just like self.output.placeholder.
    # However, in many cases, this will just be {0: time-lengths} and the same as from the input.
    # We check for this case and preset it by that if possible.
    # If you want to have it different in your layer, just overwrite it.
    if sources and sources[0].output.matches_var_dim_pattern(output):
      output.size_placeholder = sources[0].output.size_placeholder.copy()
    elif target or size_target:
      if network.train_flag is not False:
        # TODO: In training, this is ok. Maybe as well as for eval but not clear.
        # In forward, mark_data_key_as_used=False should be used and anyway that target value is not available.
        output.size_placeholder = cls._static_get_target_value(
          target=target or size_target, network=network,
          mark_data_key_as_used=network.train_flag is not False).size_placeholder.copy()
    if any([(not src.output.available_for_inference) for src in sources]):
      output.available_for_inference = False

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
    This is used by :func:`TFNetwork.construct_from_dict`.
    """
    src_names = d.pop("from", ["data"])
    if not isinstance(src_names, (list, tuple)):
      src_names = [src_names]
    d["sources"] = [
      get_layer(src_name)
      for src_name in src_names
      if not src_name == "none"]
    if "reuse_params" in d:
      d["reuse_params"] = get_layer(d["reuse_params"])
    if d.get("loss", None) and "target" not in d:
      d["target"] = network.extern_data.default_target
    if d.get("target"):
      if network.eval_flag:
        # Not resolving this in the dict, but call get_layer to make it available.
        assert isinstance(d["target"], str)
        if d["target"].startswith("layer:"):
          get_layer(d["target"][len("layer:"):])
    if "n_out" not in d and d.get("target", None):
      # Must be done here now because loss might be set to None later.
      d["n_out"] = cls._guess_n_out_from_target_and_opt_loss(
        network=network, target=d["target"], loss_class_name=d.get("loss", None))
    d["loss"] = cls._make_loss(
      class_name=d.pop("loss", None), opts=d.pop("loss_opts", None), network=network, get_layer=get_layer)

  @classmethod
  def _guess_n_out_from_target_and_opt_loss(cls, network, target, loss_class_name):
    """
    :param TFNetwork.TFNetwork network:
    :param str target: e.g. "classes"
    :param str|None loss_class_name: e.g. "ce" or None
    :return: n_out value
    :rtype: int
    """
    n_out = cls._static_get_target_value(target=target, network=network, mark_data_key_as_used=False).dim
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
    return self.cls_get_tf_scope_name(name=self.name)

  def get_absolute_name_scope_prefix(self):
    """
    :return: e.g. "output/", always with "/" at end
    :rtype: str
    """
    return self.network.get_absolute_name_scope_prefix() + self.tf_scope_name + "/"

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

  def get_dep_layers(self):
    """
    :return: list of layers this layer depends on.
      normally this is just self.sources but e.g. the attention layer in addition has a base, etc.
    :rtype: list[LayerBase]
    """
    return list(self.sources)

  def get_search_beam_size(self):
    """
    :return: beam size if there was a choice layer and we do search
    :rtype: int|None
    """
    if self.network.search_flag:
      choices = self.network.get_search_choices(src=self)
      if choices:
        return choices.search_choices.beam_size
    return None

  def get_batch_dim(self):
    """
    The batch dim by this layer, not taken from our output but calculated.
    Normally it is self.network.get_batch_dim()
    but if we do search and there was a choice layer, it it multiplied by the beam size.
    :return: batch dim * beam size
    :rtype: tf.Tensor
    """
    batch_dim = self.network.get_batch_dim()
    beam_size = self.get_search_beam_size()
    if beam_size is not None:
      batch_dim *= beam_size
    return batch_dim

  @contextlib.contextmanager
  def var_creation_scope(self):
    """
    This takes care of setting up a scope where variables can be created.

    :return: yields the variable_scope
    """
    from TFUtil import var_creation_scope, get_current_var_scope_name, reuse_name_scope
    with var_creation_scope() as dep:
      if self.reuse_params:
        cur_scope = get_current_var_scope_name()
        self_base_scope = self.get_absolute_name_scope_prefix()
        assert self_base_scope.endswith("/")
        assert (cur_scope + "/").startswith(self_base_scope)
        ext_scope = cur_scope[len(self_base_scope) - 1:]  # e.g. "/rec" or ""
        assert not ext_scope or ext_scope.startswith("/")
        reuse_base_scope = self.reuse_params.get_absolute_name_scope_prefix()
        assert reuse_base_scope.endswith("/")
        reuse_scope = reuse_base_scope[:-1] + ext_scope
        with reuse_name_scope(reuse_scope, absolute=True, reuse_vars=True) as scope:
          yield scope
      else:
        yield tf.get_variable_scope()

  def add_param(self, param, custom_update=None):
    """
    :param tf.Variable param:
    :param None|CustomUpdate custom_update: will be applied in training, instead of taking the gradient
    :return: param
    :rtype tf.Variable
    """
    assert isinstance(param, tf.Variable)
    if custom_update:
      custom_update.set_on_var(param)
    if self.reuse_params:
      name_scope_prefix = self.reuse_params.get_absolute_name_scope_prefix()
    else:
      name_scope_prefix = self.get_absolute_name_scope_prefix()
    assert param.name
    assert param.name[:len(name_scope_prefix)] == name_scope_prefix
    assert param.name[-2:] == ":0"
    param_name = param.name[len(name_scope_prefix):-2]
    self.params[param_name] = param
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
      d[param_name] = param
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
    :return: the loss, a scalar value, or None if not set. not multiplied by loss_scale
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
    assert self.output.have_time_axis()
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
        from TFUtil import constant_with_shape
        return tf.cast(constant_with_shape(v, shape=shape), dtype=data.dtype)
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


class SearchChoices(object):
  def __init__(self, owner, src_beams=None, beam_size=None, is_decided=False):
    """
    :param LayerBase owner:
    :param tf.Tensor|None src_beams: (batch, beam) -> src beam index
    :param int|None beam_size:
    :param bool is_decided: by decide layer
    """
    self.owner = owner
    self._done_src_layer = False
    self._src_layer = None  # type: LayerBase
    self.src_beams = src_beams
    self.beam_size = beam_size
    self.beam_scores = None  # type: tf.Tensor
    self.is_decided = is_decided

  def __repr__(self):
    s = " beam_size=%r" % self.beam_size
    if self._done_src_layer:
      s += " src_layer=%r" % self._src_layer
    s += " beam_scores=%r" % self.beam_scores
    if self.is_decided:
      s += " is_decided"
    return "<SearchChoices owner=%r%s>" % (self.owner, s)

  @property
  def src_layer(self):
    """
    :rtype: LayerBase
    """
    if not self._done_src_layer:
      self._src_layer = self.owner.network.get_search_choices(sources=self.owner.sources)
      self._done_src_layer = True
    return self._src_layer

  def set_beam_scores_from_own_rec(self):
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

  def filter_seqs(self, seq_filter):
    """
    :param tf.Tensor seq_filter: (batch, beam) of type bool
    """
    with tf.name_scope("search_filter_seqs"):
      from TFUtil import expand_dims_unbroadcast, get_shape_dim
      beam_size = get_shape_dim(self.beam_scores, axis=-1)
      src_layer = self.src_layer
      src_search = src_layer.search_choices
      self.beam_scores = tf.where(seq_filter, self.beam_scores, src_search.beam_scores)
      initial_src_beams = tf.range(0, beam_size, dtype=self.src_beams.dtype)  # (beam,)
      initial_src_beams = expand_dims_unbroadcast(
        initial_src_beams, axis=0, dim=get_shape_dim(self.beam_scores, axis=0))  # (batch, beam)
      self.src_beams = tf.where(seq_filter, self.src_beams, initial_src_beams)


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
    """
    :param TFNetwork.TFNetwork network:
    :param str|None data_key:
    :rtype: Data
    """
    if data_key is None:
      data_key = network.extern_data.default_input
    return network.get_extern_data(data_key, mark_data_key_as_used=False)


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
    name = "_".join([l.tf_scope_name for l in src_layers])
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
  with _name_scope_for_concat_src_layers(src_layers, "concat_sources"):
    data.placeholder = tf.concat(
      axis=len(prefix_shape) + 1,  # one more because this is with batch-dim
      values=[layer.output.get_placeholder_with_specific_batch_dim_axis(data.batch_dim_axis) for layer in src_layers])
  data.size_placeholder = src_layers[0].output.size_placeholder.copy()
  network.concat_sources_dropout_cache[(tuple(src_layers), 0.0)] = data.copy()
  return data


def get_concat_sources_data_template(src_layers, name="concat_sources"):
  """
  :param list[LayerBase] src_layers:
  :param str name: name of the Data
  :return: data with no placeholders set
  :rtype: Data
  """
  assert src_layers, "need source layers"
  dim = 0
  beam_size = None
  for layer in src_layers:
    shape = layer.output.shape
    assert shape[-1], "source last-dim must be specified of layer %r" % layer
    dim += shape[-1]
    beam_size = beam_size or layer.output.beam_size
  data = Data(
    name=name,
    shape=src_layers[0].output.shape[:-1] + (dim,),
    dim=dim,
    sparse=False,
    batch_dim_axis=src_layers[0].output.batch_dim_axis,
    time_dim_axis=src_layers[0].output.time_dim_axis,
    dtype=src_layers[0].output.dtype,
    beam_size=beam_size)
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
    return data.copy()
  if (tuple(src_layers), float(dropout)) in network.concat_sources_dropout_cache:
    return network.concat_sources_dropout_cache[(tuple(src_layers), float(dropout))].copy()
  data = data.copy()
  assert 0.0 < dropout < 1.0
  with _name_scope_for_concat_src_layers(src_layers, "dropout_in_train"):
    import TFUtil
    fn_train = lambda: TFUtil.dropout(
      data.placeholder,
      keep_prob=1 - dropout,
      # noise_shape is like old behavior for now:
      # all dynamic dimensions (batch,time) will use the same dropout-mask broadcasted.
      noise_shape=data.get_bc_spatial_batch_shape(),
      seed=network.random.randint(2 ** 31))
    fn_eval = lambda: data.placeholder
    data.placeholder = network.cond_on_train(fn_train, fn_eval)
  network.concat_sources_dropout_cache[(tuple(src_layers), float(dropout))] = data.copy()
  return data


class _ConcatInputLayer(LayerBase):
  """
  Base layer which concatenates all incoming source layers in the feature dimension,
  and provides that as `self.input_data`.
  This is the most common thing what many layers do with the input sources.
  If there is only a single source, will not do anything.
  This layer also optionally can do dropout on the input.
  """

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
    self.dropout = dropout
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
    if len(self.sources) == 1:
      self.output_loss = self.sources[0].output_loss
      if not self.dropout:
        self.output_before_activation = self.sources[0].output_before_activation

  @classmethod
  def get_out_data_from_opts(cls, name, sources=(), out_type=None, n_out=None, **kwargs):
    if out_type or n_out:
      return super(CopyLayer, cls).get_out_data_from_opts(name=name, out_type=out_type, n_out=n_out, sources=sources, **kwargs)
    return get_concat_sources_data_template(sources, name="%s_output" % name)


class InternalLayer(LayerBase):
  """
  This is not supposed to be used by the user.
  It is used by some code to construct a wrapper layer or so.
  """


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
    import inspect
    batch_norm_kwargs = inspect.getargspec(self.batch_norm).args[1:]  # first is self, ignore
    batch_norm_opts = {key: kwargs.pop(key)
                       for key in batch_norm_kwargs
                       if key in kwargs}
    super(BatchNormLayer, self).__init__(batch_norm=batch_norm_opts or True, **kwargs)


class SliceLayer(_ConcatInputLayer):
  """
  Slicing on the input, i.e. x[start:end:step] in some axis.
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
    if axis == self.input_data.time_dim_axis:
      if slice_start:
        assert slice_start > 0
        self.output.size_placeholder[self.input_data.time_dim_axis_excluding_batch] = \
          tf.maximum(0, self.output.size_placeholder[self.input_data.time_dim_axis_excluding_batch] - slice_start)
      if slice_end is not None:
        if slice_end < 0:
          slice_end = tf.shape(self.input_data.placeholder)[self.input_data.time_dim_axis] + slice_end
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
  def get_out_data_from_opts(
        cls, name, axis, sources=(),
        slice_start=None, slice_end=None, slice_step=None, **kwargs):
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
      if axis_wo_batch == len(out_type["shape"]) - 1 and not out_type["sparse"]:
        assert out_type["shape"][axis_wo_batch]
        out_type["dim"] = out_type["shape"][axis_wo_batch]
    return Data(**out_type)


class LinearLayer(_ConcatInputLayer):
  """
  Linear/forward/fully-connected/1x1-conv layer.
  Does a linear transformation on the feature-dimension of the input
  with an optional bias term and an optional activation function.
  """
  layer_class = "linear"

  def __init__(self, activation, with_bias=True, grad_filter=None,
               forward_weights_init="glorot_uniform", bias_init=0.0,
               **kwargs):
    """
    :param str|None activation: e.g. "relu", or None
    :param bool with_bias:
    :param float|None grad_filter: if grad norm is higher than this threshold (before activation), the grad is removed
    :param str forward_weights_init: see :func:`TFUtil.get_initializer`
    :param str recurrent_weights_init: see :func:`TFUtil.get_initializer`
    :param str|float bias_init: see :func:`TFUtil.get_initializer`
    """
    super(LinearLayer, self).__init__(**kwargs)
    from TFUtil import get_initializer

    self.activation = activation
    self.with_bias = with_bias

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
      W = self.add_param(tf.get_variable(
        name="W", shape=(n_in, n_out), dtype=tf.float32, initializer=fwd_weights_initializer))

      if self.with_bias:
        bias_initializer = get_initializer(
          bias_init, seed=self.network.random.randint(2 ** 31) if bias_init else 0, eval_local_ns={"layer": self})
        b = self.add_param(tf.get_variable(
          name="b", shape=(n_out,), dtype=tf.float32, initializer=bias_initializer))
      else:
        assert not bias_init
        b = None

    with tf.name_scope("linear"):
      from TFUtil import dot
      x = input_data.placeholder
      ndim = x.get_shape().ndims

      if self.input_data.sparse:
        if x.dtype in [tf.uint8, tf.int8, tf.uint16, tf.int16]:
          x = tf.cast(x, tf.int32)
        # Maybe optionally we could also use tf.contrib.layers.safe_embedding_lookup_sparse().
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
  """
  Just a LinearLayer with activation="softmax" by default.
  """
  layer_class = "softmax"

  def __init__(self, activation="softmax", **kwargs):
    super(SoftmaxLayer, self).__init__(activation=activation, **kwargs)


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
  """
  layer_class = "window"
  recurrent = True  # we must not allow any shuffling in the time-dim or so

  def __init__(self, window_size, axis="T", padding="same", **kwargs):
    """
    :param int window_size:
    :param str|int axis: see Data.get_axis_from_description()
    :param str padding: "same" or "valid"
    :param kwargs:
    """
    super(WindowLayer, self).__init__(**kwargs)
    data = self.input_data.copy_as_batch_major()
    axis = data.get_axis_from_description(axis)
    from TFUtil import windowed_nd
    self.output.placeholder = windowed_nd(
      data.placeholder,
      window=window_size, padding=padding, time_axis=axis, new_window_axis=axis + 1)
    self.output.placeholder.set_shape(tf.TensorShape(self.output.batch_shape))
    # Note: size_placeholder not correct with padding="valid" in time axis...
    self.output.size_placeholder = self.input_data.size_placeholder.copy()

  @classmethod
  def get_out_data_from_opts(cls, window_size, axis="T", sources=(), **kwargs):
    data = get_concat_sources_data_template(sources)
    data = data.copy_as_batch_major()
    axis = data.get_axis_from_description(axis)
    data.shape = data.shape[:axis] + (window_size,) + data.shape[axis:]  # add new axis right after
    return data


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
  Merges a list of axes into a single one.
  E.g. input is (batch, width, height, dim) and axes=(1,2), then we get (batch, width*height, dim).
  Or input is (batch, time, height, dim) and axes="except_time", then we get (batch, time, height*dim).
  See also CombineDimsLayer.
  """
  layer_class = "merge_dims"

  def __init__(self, axes, n_out=None, **kwargs):
    """
    :param str|list[str]|list[int] axes: see Data.get_axes_from_description(), e.g. "except_time"
    :param int|None n_out:
    """
    super(MergeDimsLayer, self).__init__(**kwargs)
    axes = self.input_data.get_axes_from_description(axes)
    if self.input_data.batch_dim_axis not in axes:
      merge_target_axis = axes[0]
    else:
      # In case we also merge the batch-dim-axis, we will merge everything into
      merge_target_axis = self.input_data.batch_dim_axis
    x = self.input_data.placeholder
    if len(axes) > 1:
      axes = sorted(axes)
      # Transpose so that all axes are behind each other.
      perm = list(range(self.input_data.batch_ndim))
      i0 = merge_target_axis
      for i, a in enumerate([a for a in axes if a != i0]):
        perm.remove(a)
        if a < i0:
          i0 -= 1
        perm.insert(i0 + i + 1, a)
      x = tf.transpose(x, perm)
      # Now merge all dims with a reshape.
      shape = tf.shape(x)
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
    self.output.size_placeholder = self._get_output_sizes(axes=axes, target_axis=merge_target_axis)

  def _get_output_sizes(self, axes, target_axis):
    """
    :param list[int] axes:
    :param int target_axis:
    """
    removed_axes = [a for a in axes if a != target_axis]
    assert len(removed_axes) == len(axes) - 1
    d = {}
    for i, v in sorted(self.input_data.size_placeholder.items()):
      axis = self.input_data.get_batch_axis(i)
      if axis in axes and axis != target_axis:
        continue
      axis -= len([a for a in axes if a < axis])
      if axis == self.input_data.batch_dim_axis:
        continue
      j = self.input_data.get_batch_axis_excluding_batch(axis)
      if j in d:
        d[j] *= v
      else:
        d[j] = v
    return d

  @classmethod
  def get_out_data_from_opts(cls, name, axes, sources=(), n_out=None, out_type=None, **kwargs):
    assert not out_type, "currently ignored"
    data = get_concat_sources_data_template(sources)
    data.name = "%s_output" % name
    axes = data.get_axes_from_description(axes)
    if len(axes) <= 1:
      return data
    axes = sorted(axes)
    import numpy
    new_shape = list(data.shape)
    res_dim = None
    if all([data.batch_shape[i] is not None for i in axes]):
      res_dim = numpy.prod([data.batch_shape[i] for i in axes])
    if not data.sparse and data.feature_dim_axis in axes:  # will also merge the feature dim
      assert axes == list(range(axes[0], data.batch_ndim)), "not supported currently with holes"
      if res_dim is not None and n_out is not None:
        assert res_dim == n_out
      elif res_dim is not None and n_out is None:
        pass
      elif res_dim is None and n_out is not None:
        res_dim = n_out
      else:
        raise Exception(
          "You need to provide n_out for layer %r, we are merging axes %r with dims %r. Input is %r." % (
          name, axes, [data.batch_shape[i] for i in axes], data))
      data.dim = res_dim
    if data.batch_dim_axis not in axes:
      merge_target_axis = axes[0]
      new_shape[data.get_batch_axis_excluding_batch(merge_target_axis)] = res_dim
    else:
      # In case we also merge the batch-dim-axis, we will merge everything into
      merge_target_axis = data.batch_dim_axis
    for i in reversed(axes):
      if i == merge_target_axis:
        continue
      if i == data.batch_dim_axis:
        continue
      new_shape.pop(data.get_batch_axis_excluding_batch(i))
      if data.batch_dim_axis >= i:
        data.batch_dim_axis -= 1
      if data.time_dim_axis is not None and data.time_dim_axis >= i:
        data.time_dim_axis -= 1
    data.shape = tuple(new_shape)
    return data


class SplitDimsLayer(_ConcatInputLayer):
  """
  Splits one axis into multiple axes.
  E.g. if you know that your feature-dim is composed by a window,
  i.e. the input is (batch, time, window * feature),
  you can set axis="F", dims=(window, -1),
  and you will get the output (batch, time, window, feature).
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
    data = get_concat_sources_data_template(sources)
    data.name = "%s_output" % name
    if isinstance(axis, int):
      data = data.copy_as_batch_major()
    axis = data.get_axis_from_description(axis)
    assert axis != data.batch_dim_axis
    if data.batch_shape[axis] is not None:
      resolved_shape_dims = cls._resolve_dims(old_dim=data.batch_shape[axis], new_dims=dims)
    else:
      resolved_shape_dims = tuple([(d if (d >= 0) else None) for d in dims])
    if axis == data.feature_dim_axis:
      data.dim = cls._resolve_dims(old_dim=data.dim, new_dims=dims)[-1]
    axis_wb = data.get_batch_axis_excluding_batch(axis)
    data.shape = data.shape[:axis_wb] + resolved_shape_dims + data.shape[axis_wb + 1:]
    return data


class SplitBatchTimeLayer(_ConcatInputLayer):
  """
  A very specific layer which expects to get input of shape (batch * time, ...)
  and converts it into (batch, time, ...), where it recovers the seq-lens from some other layer.
  """
  layer_class = "split_batch_time"

  def __init__(self, base, **kwargs):
    """
    :param LayerBase base:
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
    super(SplitBatchTimeLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["base"] = get_layer(d.get("base", "data"))

  @classmethod
  def get_out_data_from_opts(cls, name, base, sources=(), **kwargs):
    data = get_concat_sources_data_template(sources)
    data.name = "%s_output" % name
    assert data.batch_dim_axis == 0
    data.time_dim_axis = 1
    data.shape = (None,) + data.shape
    return data


class ExpandDimsLayer(_ConcatInputLayer):
  """
  Adds some axis.
  """
  layer_class = "expand_dims"

  def __init__(self, axis, dim=1, **kwargs):
    """
    :param str|int axis: axis to add, e.g. "F"|"feature" or "spatial".
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
    :param str axis: e.g. "F"|"feature" or "spatial"
    :return: axis as int for data.placeholder
    :rtype: int
    """
    if isinstance(axis, int):
      return axis
    assert isinstance(axis, str)
    axis = axis.lower()
    if axis in ["f", "feature"]:
      assert not data.sparse
      return data.batch_ndim
    elif axis == "spatial":
      if data.sparse:
        return data.batch_ndim
      else:
        return data.batch_ndim - 1
    else:
      raise Exception("invalid axis %r" % axis)

  @classmethod
  def get_out_data_from_opts(cls, name, axis, dim=1, sources=(), **kwargs):
    data = get_concat_sources_data_template(sources)
    data.name = "%s_output" % name
    if isinstance(axis, int):
      data = data.copy_as_batch_major()
    axis = cls._get_axis(data=data, axis=axis)
    if axis == data.batch_ndim and not data.sparse:
      data.dim = dim
    data.shape = data.shape[:axis] + (dim,) + data.shape[axis:]
    return data


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
    dyn_axes = self.input_data.get_spatial_axes()  # conv-dims, or also called spatial dims
    static_axes = self.input_data.get_feature_axes()  # feature-dims
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
    from TFUtil import get_initializer
    with self.var_creation_scope():
      fwd_weights_initializer = get_initializer(
        forward_weights_init, seed=self.network.random.randint(2 ** 31), eval_local_ns={"layer": self})
      filters = self.add_param(tf.get_variable(name="W", shape=filter_shape, initializer=fwd_weights_initializer))
    y = tf.nn.convolution(x, filter=filters, padding=padding, strides=strides, dilation_rate=dilation_rate)
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
      i: self.input_data.size_placeholder[i]
      for i in dyn_axes
      if i in self.input_data.size_placeholder}
    for i in list(self.output.size_placeholder.keys()):
      self.output.size_placeholder[i] = self.calc_out_dim(
        in_dim=self.output.size_placeholder[i],
        filter_size=filter_size[i], stride=strides[i], dilation_rate=dilation_rate[i], padding=padding)

  @classmethod
  def calc_out_dim(cls, in_dim, filter_size, stride, padding, dilation_rate=1):
    """
    :param int|tf.Tensor in_dim: dimension in some axis
    :param int filter_size: e.g. 2, for the corresponding axis
    :param int stride: e.g. 1, for the corresponding axis
    :param int dilation_rate: e.g. 1
    :param str padding: "valid" or "same"
    :return: the output dimension
    :rtype: int
    """
    def ceildiv(a, b):
      return -(-a // b)
    padding = padding.upper()
    # See tf.nn.convolution() documentation for more.
    if padding == "SAME":
      return ceildiv(in_dim, stride)
    elif padding == "VALID":
      return ceildiv((in_dim - (filter_size - 1) * dilation_rate), stride)
    else:
      raise Exception("invalid padding %r" % padding)

  @classmethod
  def _get_out_type_from_opts(cls, name, n_out, filter_size, padding, strides=1, dilation_rate=1, sources=(),
                              input_expand_dims=0, input_add_feature_dim=False, input_split_feature_dim=None, **kwargs):
    shape = [None] * len(filter_size) + [n_out]
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
    padding = padding.upper()
    if input_expand_dims == 0 and not input_add_feature_dim and not input_split_feature_dim:
      # Maybe we have a chance to correctly define the output shapes.
      data = get_concat_sources_data_template(sources)
      for i in range(len(filter_size)):
        if data.shape[i] is not None:
          shape[i] = cls.calc_out_dim(
            in_dim=data.shape[i],
            filter_size=filter_size[i], stride=strides[i], dilation_rate=dilation_rate[i], padding=padding)
    return {
      "dim": n_out,
      "shape": shape,
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
      i: self.input_data.size_placeholder[i]
      for i in range(len(pool_size))
      if i in self.input_data.size_placeholder}
    for i in list(self.output.size_placeholder.keys()):
      self.output.size_placeholder[i] = ConvLayer.calc_out_dim(
        in_dim=self.output.size_placeholder[i],
        filter_size=pool_size[i], stride=strides[i], dilation_rate=dilation_rate[i], padding=padding)

  @classmethod
  def get_out_data_from_opts(cls, name, pool_size, strides=None, dilation_rate=1, sources=(), padding="VALID", **kwargs):
    # y shape is [batch] + spatial_dims + [n_out].
    data = get_concat_sources_data_template(sources)
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
    padding = padding.upper()
    for i in range(len(pool_size)):
      if data.shape[i] is not None:
        shape[i] = ConvLayer.calc_out_dim(
          in_dim=data.shape[i],
          filter_size=pool_size[i], stride=strides[i], dilation_rate=dilation_rate[i], padding=padding)
    return Data(
      name="%s_output" % name,
      shape=tuple(shape),
      dim=data.dim,
      dtype=data.dtype,
      sparse=False,
      batch_dim_axis=0,
      beam_size=data.beam_size)


class ReduceLayer(_ConcatInputLayer):
  """
  This reduces some axis by using "sum" or "max".
  It's basically a wrapper around tf.reduce_sum or tf.reduce_max.
  """
  layer_class = "reduce"

  def __init__(self, mode, axes=None, axis=None, keep_dims=False, enforce_batch_dim_axis=None, **kwargs):
    """
    :param str mode: "sum" or "max" or "mean"
    :param int|list[int]|str axes: one axis or multiple axis to reduce.
      this is counted with batch-dim, which by default is axis 0 (see enforce_batch_dim_axis).
      it also accepts the special tokens "B"|"batch", "spatial", "spatial_except_time", or "F"|"feature"
    :param int|list[int]|str axis: for compatibility, can be used instead of ``axes``
    :param bool keep_dims: if dimensions should be kept (will be 1)
    :param int enforce_batch_dim_axis: will swap the batch-dim-axis of the input with the given axis.
      e.g. 0: will convert the input into batch-major format if not already like that.
    """
    if axis is not None:
      print("reduce layer %r: option 'axis' is deprecated, use 'axes' instead" % kwargs["name"], file=log.v4)
      assert axes is None, "don't provide both 'axes' and 'axis', layer %r" % kwargs["name"]
      axes = axis
    if enforce_batch_dim_axis is None and self.need_enforce_batch_dim_axis(axes):
      enforce_batch_dim_axis = 0
    assert "n_out" not in kwargs
    assert "out_type" not in kwargs
    mode = mode.lower()
    assert mode in ["max", "sum", "avg", "mean"]
    super(ReduceLayer, self).__init__(**kwargs)
    assert not self.input_data.sparse
    x = self.input_data
    if enforce_batch_dim_axis is not None and x.batch_dim_axis != enforce_batch_dim_axis:
      x = x.copy_with_batch_dim_axis(enforce_batch_dim_axis)
    axes = self.get_axes(axes, input_data=x)
    if mode == "max":
      f = tf.reduce_max
    elif mode == "sum":
      f = tf.reduce_sum
    elif mode in ["avg", "mean"]:
      f = tf.reduce_mean
    else:
      assert False
    if x.time_dim_axis in axes:
      assert not keep_dims, "not yet implemented otherwise"
      assert x.batch_dim_axis in axes, "not yet implemented otherwise"
      axes = [a if (a < x.time_dim_axis) else (a - 1)
              for a in axes if a != x.time_dim_axis]
      x = x.copy_time_flattened()
    y = f(x.placeholder, axis=axes, keep_dims=keep_dims)
    y_dyn_sizes = x.size_placeholder.copy()
    if keep_dims:
      for i in axes:
        if i in y_dyn_sizes:
          y_dyn_sizes[i] = 1
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
    if keep_dims:
      for i in axes:
        y_shape[i] = 1
      y_shape.remove(x.batch_dim_axis)
    else:
      for i in reversed(sorted(set(axes + [x.batch_dim_axis]))):
        del y_shape[i]
    return Data(
      name="%s_output" % name,
      shape=y_shape,
      batch_dim_axis=x.batch_dim_axis if (x.batch_dim_axis not in axes) else None,
      dtype=x.dtype,
      sparse=False,
      beam_size=x.beam_size)


class SqueezeLayer(_ConcatInputLayer):
  """
  Removes an axis with dimension 1.
  This is basically a wrapper around tf.squeeze.
  """
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


class WeightedSumLayer(_ConcatInputLayer):
  """
  Calculates a weighted sum, either over a complete axis of fixed dimension, or over some window.
  Can also do that for multiple axes.
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
    # Just the same as the input.
    return get_concat_sources_data_template(sources, name="%s_output" % name)


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
  """
  Resizes the input, i.e. upsampling or downsampling.
  Supports different kinds, such as linear interpolation or nearest-neighbor.
  """
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
  """
  Combines multiple dimensions.
  See also MergeDimsLayer.
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


class FsaLayer(LayerBase):
  layer_class = "fsa"

  def __init__(self, **kwargs):
    """
    """
    super(FsaLayer, self).__init__(**kwargs)
    # TODO...


class CombineLayer(LayerBase):
  """
  Applies some binary operation on all sources, such as addition.
  """
  layer_class = "combine"

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
    assert kind in ["average", "add", "sub", "mul", "eval"], \
        "Invalid `kind` %r for this layer." % kind
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
  def get_out_data_from_opts(cls, n_out=None, out_type=None, sources=(), **kwargs):
    if not n_out and not out_type:
      out_type = sources[0].output.get_kwargs()
      out_type["name"] = "%s_output" % kwargs["name"]
    return super(CombineLayer, cls).get_out_data_from_opts(n_out=n_out, out_type=out_type, sources=sources, **kwargs)

  def _check_same_dense_dim(self, sources):
    """
    :param list[LayerBase] sources:
    """
    assert not self.output.sparse
    for source in sources:
      assert not source.output.sparse
      assert source.output.dim == self.output.dim \
              or source.output.dim == 1 # Constant layer broadcasting

  # Requires the same input shape and yield the same output shape.
  def _op_dense_fn(self, sources, fn):
    """
    :param list[LayerBase] sources:
    :param ((x1,x2) -> y) fn: function to perform on x1 and x2
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
    used_sources = set()  # type: set[int]
    def source(i):
      """
      :param int i: layer index
      :return: output placeholder from source i, compatible to source 0
      :rtype: tf.Tensor
      """
      assert 0 <= i < len(sources)
      used_sources.add(i)
      if isinstance(sources[i], LayerBase):
        if i == 0:
          return sources[i].output.placeholder
        return sources[i].output.copy_compatible_to(sources[0].output).placeholder
      return sources[i]
    vs = vars(TFUtil).copy()
    vs.update({"tf": tf, "source": source, "self": self})
    vs.update(eval_locals or {})
    x = eval(eval_str, vs)
    assert sorted(used_sources) == list(range(len(sources))), (
      "not used sources: %r" % set(range(len(sources))).difference(used_sources))
    return x

  def _get_op(self, kind, eval_str=None, eval_locals=None):
    op = getattr(self, "_op_kind_%s" % kind)
    if eval_str:
      assert kind == "eval"
      def wrap_eval_op(sources):
        return self._op_kind_eval(sources, eval_str=eval_str, eval_locals=eval_locals)
      op = wrap_eval_op
    return op


class EvalLayer(CombineLayer):
  """
  Evaluates some string.
  The CombineLayer provides this functionality, thus this is just a special case of it.
  """
  layer_class = "eval"

  def __init__(self, **kwargs):
    super(EvalLayer, self).__init__(kind="eval", **kwargs)


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
  def get_out_data_from_opts(cls, n_out=None, out_type=None, sources=(), **kwargs):
    if not n_out and not out_type:
      out_type = sources[0].output.get_kwargs()
      out_type["name"] = "%s_output" % kwargs["name"]
      if out_type.get("sparse", False):
        out_type["dim"] = 2  # True or False
      out_type["dtype"] = "bool"
    return super(CompareLayer, cls).get_out_data_from_opts(n_out=n_out, out_type=out_type, sources=sources, **kwargs)


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
      name="%s/%s:subnet" % (self.network.name, self.name),
      rnd_seed=self.network.random.randint(2**31),
      train_flag=self.network.train_flag,
      extern_data=sub_extern_data,
      parent_layer=self)
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
    v = self.subnetwork.get_total_constraints()
    if v is 0:
      return None
    return v

  def get_loss_value(self):
    v = self.subnetwork.get_total_loss()
    if v is 0:
      return None
    return v

  def get_error_value(self):
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


class AccumulateMeanLayer(ReduceLayer):
  """
  Accumulates the mean of the input (in training).
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
    return super(AccumulateMeanLayer, cls).get_out_data_from_opts(axes=axes, **kwargs)


class FastBaumWelchLayer(_ConcatInputLayer):
  """
  Calls :func:`fast_baum_welch` or :func:`fast_baum_welch_by_sprint_automata`.
  We expect that our input are +log scores.
  """
  layer_class = "fast_bw"
  recurrent = True

  def __init__(self, align_target, sprint_opts=None, **kwargs):
    """
    :param str align_target: e.g. "sprint"
    :param dict[str] sprint_opts:
    """
    super(FastBaumWelchLayer, self).__init__(**kwargs)
    assert align_target == "sprint", "not yet implemented otherwise, align_target %r" % align_target
    data = self.input_data.copy_as_time_major()
    from TFUtil import sequence_mask_time_major
    seq_mask = sequence_mask_time_major(data.get_sequence_lengths())
    from TFNativeOp import fast_baum_welch_by_sprint_automata
    seq_tags = self.network.get_seq_tags()
    fwdbwd, obs_scores = fast_baum_welch_by_sprint_automata(
      sprint_opts=sprint_opts,
      am_scores=-data.placeholder,  # it wants the scores in -log space
      float_idx=seq_mask,
      tags=seq_tags)
    loss = tf.reduce_sum(obs_scores[0])
    self.output_loss = loss
    bw = tf.exp(-fwdbwd)
    self.output.placeholder = bw
    self.output.size_placeholder = data.size_placeholder.copy()

  @classmethod
  def get_out_data_from_opts(cls, name, sources, **kwargs):
    return get_concat_sources_data_template(sources, name="%s_output" % name).copy_as_time_major()


class SyntheticGradientLayer(_ConcatInputLayer):
  """
  This is a generalized way to be able to replace the true gradient with any kind of predicted gradient.
  This enabled to implement the idea from here:
    Decoupled Neural Interfaces using Synthetic Gradients, https://arxiv.org/abs/1608.05343
  """
  layer_class = "synthetic_gradient"

  def __init__(self, gradient, **kwargs):
    """
    :param LayerBase gradient:
    """
    super(SyntheticGradientLayer, self).__init__(**kwargs)
    from TFUtil import SyntheticGradient
    self.output.placeholder = SyntheticGradient.synthetic_gradient(
      x=self.input_data.placeholder,
      synthetic_grad_x=gradient.output.copy_compatible_to(self.input_data).placeholder)
    self.output.size_placeholder = self.input_data.size_placeholder.copy()

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    super(SyntheticGradientLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["gradient"] = get_layer(d["gradient"])

  @classmethod
  def get_out_data_from_opts(cls, sources, name, **kwargs):
    return get_concat_sources_data_template(sources, name="%s_output" % name)


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
    result = [None] * self.output.dim
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
  def get_out_data_from_opts(cls, name, sources, context_len=1, n_out=None, **kwargs):
    assert len(sources) == 1, "%s: We expect exactly one source layer." % name
    dim = 3 + context_len * 2  # (center, left_1, right_1, ..., state, boundary)
    if n_out is not None:
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
    output_before_softmax_flat = flatten_with_seq_len_mask(source.output_before_activation.x, output_seq_lens, time_major=output.is_time_major)
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
        with self.var_creation_scope():
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
    return Data(name="framewise_statistics_dummy_output", shape=(), dtype="int32", batch_dim_axis=None)


class Loss(object):
  """
  Base class for all losses.
  """
  class_name = None  # type: str  # used by get_loss_class()
  recurrent = False  # if this is a frame-wise criteria, this will be False

  def __init__(self, base_network):
    """
    :param TFNetwork.TFNetwork base_network:
    """
    self.base_network = base_network
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

  def init(self, output, output_with_activation=None, target=None):
    """
    :param Data output: generated output
    :param OutputWithActivation|None output_with_activation:
    :param Data target: reference target from dataset
    """
    from TFUtil import flatten_with_seq_len_mask
    with tf.name_scope("loss_init"):
      if target:
        if output.beam_size:
          if target.beam_size != output.beam_size:
            target = target.copy_extend_with_beam(output.beam_size)
        else:
          assert not target.beam_size
      self.output = output
      self.output_with_activation = output_with_activation
      self.output_seq_lens = output.size_placeholder[0]
      self.target = target
      self.target_seq_lens = target.size_placeholder[0]
      # Flat variants are with batch,time collapsed into one, masked via seq_lens.
      self.output_flat = None
      self.output_before_softmax_flat = None
      if self.output.have_time_axis():
        self.loss_norm_factor = 1.0 / tf.cast(tf.reduce_sum(self.target_seq_lens), tf.float32)
        time_and_batch_dims = (self.output.time_dim_axis, self.output.batch_dim_axis)
        assert time_and_batch_dims in [(0, 1), (1, 0)], "output time-batch-dim unexpected: %s" % self.output
        if output_with_activation and output_with_activation.act_func is tf.nn.softmax:
          self.output_before_softmax_flat = flatten_with_seq_len_mask(output_with_activation.x, self.output_seq_lens, time_major=output.is_time_major)
        else:
          self.output_flat = flatten_with_seq_len_mask(output.placeholder, self.output_seq_lens, time_major=output.is_time_major)
          self.output_flat.set_shape(tf.TensorShape(output.shape))
      else:  # no time axis
        self.loss_norm_factor = 1.0
        assert self.output.batch_ndim == 2
        if output_with_activation and output_with_activation.act_func is tf.nn.softmax:
          self.output_before_softmax_flat = output_with_activation.x
        else:
          self.output_flat = output
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
        target_label = tf.cast(tf.argmax(target_flat, axis=last_dim), tf.int32)
      output_label = tf.cast(tf.argmax(output_flat, axis=last_dim), target_label.dtype)
      not_equal = tf.not_equal(output_label, target_label)
      return self.reduce_func(tf.cast(not_equal, tf.float32))

  def get_value(self):
    """
    :return: loss as a scalar value
    :rtype: tf.Tensor|None
    """
    raise NotImplementedError

  def get_normalization_factor(self):
    """
    :return: factor as a float scalar, usually 1.0 / num_frames. see self.reduce_func.
    :rtype: tf.Tensor
    """
    return self.loss_norm_factor

  @classmethod
  def get_auto_output_layer_dim(cls, target_dim):
    """
    :param int target_dim:
    :return: normally just the same as target_dim. e.g. for CTC, we would add 1 for the blank label
    :rtype: int
    """
    return target_dim


class CrossEntropyLoss(Loss):
  """
  Cross-Entropy loss. Basically sum(target * log(output)).
  """
  class_name = "ce"

  def get_value(self):
    with tf.name_scope("loss_ce"):
      log_clip_values = (1e-32, 1e32)  # only used for non-fused path
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
          out = tf.log(tf.clip_by_value(tf.gather_nd(self.output_flat, target_flat_exp), *log_clip_values))
          return -self.reduce_func(out)
      else:  # not sparse
        if self.output_before_softmax_flat is not None:
          out = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_before_softmax_flat, labels=self.target_flat)
          return self.reduce_func(out)
        else:
          out = self.target_flat * tf.log(tf.clip_by_value(self.output_flat, *log_clip_values))
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

  def __init__(self, target_collapse_repeated=False, auto_clip_target_len=False, **kwargs):
    """
    :param bool target_collapse_repeated: like preprocess_collapse_repeated option for CTC. used for sparse_labels().
    :param bool auto_clip_target_len: see self._get_target_sparse_labels().
    """
    super(CtcLoss, self).__init__(**kwargs)
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

  @classmethod
  def get_auto_output_layer_dim(cls, target_dim):
    return target_dim + 1  # one added for blank


class EditDistanceLoss(Loss):
  """
  Note that this loss is not differentiable, thus it's only for keeping statistics.
  """
  class_name = "edit_distance"
  recurrent = True

  def __init__(self, debug_print=False, **kwargs):
    super(EditDistanceLoss, self).__init__(**kwargs)
    self._output_sparse_labels = None
    self._target_sparse_labels = None
    self._debug_print = debug_print

  def init(self, output, output_with_activation=None, target=None):
    """
    :param Data output: generated output
    :param OutputWithActivation|None output_with_activation:
    :param Data target: reference target from dataset
    """
    super(EditDistanceLoss, self).init(
      output=output, output_with_activation=output_with_activation, target=target)
    assert output.sparse
    assert target.sparse
    assert output.shape == target.shape
    assert output.dim == target.dim
    self._output_sparse_labels = None
    self._target_sparse_labels = None

  def _get_output_sparse_labels(self):
    if self._output_sparse_labels is not None:
      return self._output_sparse_labels
    from TFUtil import sparse_labels
    labels = sparse_labels(self.output.get_placeholder_as_batch_major(), seq_lens=self.output_seq_lens)
    self._output_sparse_labels = labels
    return labels

  def _get_target_sparse_labels(self):
    if self._target_sparse_labels is not None:
      return self._target_sparse_labels
    from TFUtil import sparse_labels
    labels = sparse_labels(self.target.get_placeholder_as_batch_major(), seq_lens=self.target_seq_lens)
    self._target_sparse_labels = labels
    return labels

  def _debug_print_out(self):
    def get_first_seq(data):
      x = data.get_placeholder_as_batch_major()
      seq = x[0][:data.size_placeholder[0][0]]
      return seq
    output = get_first_seq(self.output)
    target = get_first_seq(self.target)
    from TFUtil import encode_raw
    return ["output", tf.size(output), encode_raw(output), "target", tf.size(target), encode_raw(target)]

  def get_error(self):
    output = self._get_output_sparse_labels()
    labels = self._get_target_sparse_labels()
    error = tf.edit_distance(hypothesis=output, truth=labels, normalize=False)
    if self._debug_print:
      error = tf.Print(error, self._debug_print_out(), summarize=10)
    return self.reduce_func(error)

  def get_value(self):
    return None


class BinaryCrossEntropy(Loss):
  """
  """
  class_name = "bin_ce"

  def get_value(self):
    """
    """
    assert not self.target.sparse, "sparse is not supported yet"
    with tf.name_scope("loss_bin_ce"):
      out = 0.5 * tf.nn.sigmoid_cross_entropy_with_logits(logits = self.output_flat, labels = self.target_flat)
      return tf.reduce_mean(out)

class DeepClusteringLoss(Loss):
  """
  cost function used for deep clustering as described in
  [Hershey & Chen+, 2016]: "Deep clustering discriminative embeddings for segmentation and separation"
  """
  class_name = "deep_clustering"

  def __init__(self, embedding_dimension, nr_of_sources, **kwargs):
    """
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
      "Number of dimensions missmatch. Target: %s, output: %s" % (self.target, self.output))
    expected_output_dim = self._embedding_dimension * ( self.target.shape[1] / self._nr_of_sources)
    assert expected_output_dim == self.output.dim, (
      "Expected output dim is %i but the output has dim %i. " % (expected_output_dim, self.output.dim) +
      "Target: %s, output: %s" % (self.target, self.output))

  def get_error(self):
    """
    :return: frame error rate as a scalar value
    :rtype: tf.Tensor | None
    """
    return None

  def get_value(self):
    """
    """
    assert not self.target.sparse, "sparse is not supported yet"
    with tf.name_scope("loss_deep_clustering"):
      # iterate through all chunks and compute affinity cost function for every chunk separately
      c = tf.Variable([0], dtype=tf.float32)
      def iterateSequences(s, start, c):
        return tf.less(s, tf.shape(self.output_seq_lens)[0])
      def computeCost(s, start, c):
        seqLength = self.output_seq_lens[s]
        chunkOut = self.output_flat[start:(start + seqLength), :]
        chunkTarget = self.target_flat[start:(start + seqLength), :]
        # convert network output into embedding vectors 
        v = tf.reshape(tf.reshape(chunkOut, (tf.shape(chunkOut)[0], tf.shape(chunkOut)[1] / self._embedding_dimension, self._embedding_dimension)), (tf.shape(chunkOut)[0] * (tf.shape(chunkOut)[1] / self._embedding_dimension ), self._embedding_dimension))
        # convert targets into class vectors 
        y = tf.reshape(tf.reshape(chunkTarget, (tf.shape(chunkTarget)[0], tf.shape(chunkTarget)[1] / self._nr_of_sources, self._nr_of_sources)), (tf.shape(chunkTarget)[0] * (tf.shape(chunkTarget)[1] / self._nr_of_sources), self._nr_of_sources))
        chunkC = tf.pow(tf.norm(tf.matmul(tf.transpose(v), v)), 2) - 2 * tf.pow(tf.norm(tf.matmul(tf.transpose(v), y)), 2) + tf.pow(tf.norm(tf.matmul(tf.transpose(y), y)), 2)
        # append chunk cost to cost tensor
        c = tf.cond(tf.greater(s, 0), lambda: tf.concat([c, tf.reshape(chunkC, (1,))], axis=0), lambda: tf.reshape(chunkC, (1,)))
        return tf.add(s, 1), tf.add(start, seqLength), c 
      s = tf.constant(0, dtype=tf.int32) 
      start = tf.constant(0, dtype=tf.int32) 
      r = tf.while_loop(iterateSequences, computeCost, [s, start, c], shape_invariants=[s.get_shape(), start.get_shape(), tf.TensorShape([None,])])
      return tf.reduce_mean(r[-1])


class L1Loss(Loss):
  """
  L1-distance loss. sum(target - output).
  """
  class_name = "l1"

  def get_value(self):
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
    """
    assert not self.target.sparse, "sparse is not supported yet"
    with tf.name_scope("loss_mse"):
      out = tf.reduce_mean(tf.squared_difference(self.output_flat, self.target_flat))
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
    with tf.name_scope("ViaLayerLoss"):
      if self.error_signal_layer:
        assert not self.align_layer
        error_signal = self.error_signal_layer.output.copy_compatible_to(self.output).placeholder
      else:
        assert self.align_layer
        error_signal = self.output.placeholder - self.align_layer.output.copy_compatible_to(self.output).placeholder
      error_signal *= tf.cast(self.output.get_sequence_mask_broadcast(), dtype=tf.float32)
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
    if self.target is None:
      return None  # we don't have it
    # Use default frame-wise error to reference target.
    return super(ViaLayerLoss, self).get_error()


_LossClassDict = {}  # type: dict[str,type(Loss)]

def _init_loss_class_dict():
  for v in globals().values():
    if isinstance(v, type) and issubclass(v, Loss) and v.class_name:
      assert v.class_name not in _LossClassDict
      _LossClassDict[v.class_name] = v
  for alias, v in {"sse_sigmoid": BinaryCrossEntropy}.items():
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


_LayerClassDict = {}  # type: dict[str,type(LayerBase)]

def _init_layer_class_dict():
  import TFNetworkRecLayer
  for v in list(globals().values()) + list(vars(TFNetworkRecLayer).values()):
    if isinstance(v, type) and issubclass(v, LayerBase) and v.layer_class:
      assert v.layer_class not in _LayerClassDict
      _LayerClassDict[v.layer_class] = v
  for alias, v in {"forward": LinearLayer, "hidden": LinearLayer}.items():
    assert alias not in _LayerClassDict
    _LayerClassDict[alias] = v


def get_layer_class(name):
  """
  :param str name: matches layer_class
  :rtype: (() -> LayerBase) | type[LayerBase] | LayerBase
  """
  if not _LayerClassDict:
    _init_layer_class_dict()
  if name not in _LayerClassDict:
    raise Exception("unknown layer class %r" % name)
  return _LayerClassDict[name]

