
import tensorflow as tf


class Data(object):
  """
  This class is to describe a tensor,
  i.e. it's shape and properties like
  whether we should consider it as sparse data (i.e. it represents indices).
  This is used in TFNetwork to describe the dataset external data
  as well as for every layer output.
  """

  size_dtype = "int32"

  def __init__(self, name,
               shape=None, dtype=None,
               placeholder=None,
               sparse=None,
               dim=None,
               size_placeholder=None,
               auto_create_placeholders=False):
    """
    :param str name:
    :param tuple[int|None] shape: including time-dim, which can be None. excluding batch-dim
    :param str dtype: e.g. "float32" or "int64"
    :param tf.Tensor|None placeholder: with added batch-dim
    :param bool sparse: whether to treat the value as an index. do not confuse with tf.SparseTensor
    :param None|int dim: shape[-1] if not sparse, otherwise like num_classes
    :param dict[int,tf.Tensor] tf.Tensor size_placeholder: for every None in shape, this will describe the size
    """
    self.name = name
    if sparse is None:
      if dtype and dtype.startswith("int"):
        sparse = True
      elif name == "classes":
        sparse = True
      elif shape is not None and len(shape) == 1:
        sparse = True
      if sparse is None:
        sparse = False
    self.sparse = sparse
    if shape is None:
      assert dim, "no shape specified, need dim"
      if sparse:
        shape = (None,)  # assume common (time,)
      else:
        shape = (None, dim)  # assume common (time,dim)
    self.shape = shape
    if dtype is None:
      if sparse:
        dtype = "int32"
      else:
        dtype = "float32"
    if dim is None:
      assert not sparse, "need dim"
      dim = shape[-1]
    self.dim = dim
    self.dtype = dtype
    if placeholder is None and auto_create_placeholders:
      with tf.name_scope("extern_data/placeholders/%s/" % name):
        placeholder = tf.placeholder(name=name, dtype=dtype, shape=self.batch_shape)
    self.placeholder = placeholder
    if size_placeholder is None and auto_create_placeholders:
      size_placeholder = {}  # type: dict[int,tf.Tensor]
      with tf.name_scope("extern_data/placeholders/%s/" % name):
        for i, dim in enumerate(shape):
          if dim is None:
            # For each batch a separate size.
            size_placeholder[i] = tf.placeholder(
              name="%s_dim%i_size" % (name, i), dtype=self.size_dtype, shape=(None,))
    self.size_placeholder = size_placeholder

  def get_kwargs(self):
    keys = ["name", "shape", "dtype", "sparse", "dim"]
    return {key: getattr(self, key) for key in keys}

  def get_variable_dim_pattern(self):
    return tuple([dim is None for dim in self.shape])

  def get_description(self, with_name=True, with_placeholder=False):
    keys = ["shape", "dtype", "sparse"]
    if with_name:
      keys.insert(0, "name")
    if with_placeholder:
      keys.append("placeholder")
    return "Data(%s)" % ", ".join(["%s=%r" % (key, getattr(self, key)) for key in keys])

  @property
  def batch_shape(self):
      return (None,) + self.shape


class LayerBase(object):
  layer_class = None
  recurrent = False

  def __init__(self, name, network, n_out=None, out_type=None, sources=(), target=None, loss=None, L2=None, is_output_layer=None):
    """
    :param str name:
    :param TFNetwork.TFNetwork network:
    :param None|int n_out: output dim
    :param dict[str] out_type: kwargs for Data class. more explicit than n_out.
    :param list[LayerBase] sources:
    :param str|None target: if some loss is set, this is the target data-key, i.e. network.extern_data.get_data(target)
    :param str|None loss: if set, via get_loss
    :param float|None L2: for constraints
    :param bool|None is_output_layer:
    """
    self.name = name
    self.network = network
    if loss and not target:
      target = self.network.extern_data.default_target
    self.target = target
    if out_type is None and n_out is None and target:
      n_out = self.network.extern_data.get_data(target).dim
    if out_type is None:
      assert n_out
      out_type = {"dim": n_out}
    out_type = out_type.copy()
    out_type.setdefault("name", "%s_output" % self.name)
    if n_out is not None:
      out_type.setdefault("dim", n_out)
      assert out_type["dim"] == n_out
    self.output = Data(**out_type)
    # You are not supposed to set self.output.placeholder to the value which you want to return by the layer.
    # Normally you are also supposed to set self.output.size_placeholder explicitly, just like self.output.placeholder.
    # However, in many cases, this will just be {0: time-lengths} and the same as from the input.
    # We check for this case and preset it by that if possible.
    # If you want to have it different in your layer, just overwrite it.
    if sources and sources[0].output.get_variable_dim_pattern() == self.output.get_variable_dim_pattern():
      self.output.size_placeholder = sources[0].output.size_placeholder.copy()
    self.output_before_softmax = None  # type: None|tf.Tensor
    self.sources = sources
    self.params = {}  # type: dict[str,tf.Variable]
    self.loss = None
    if loss:
      loss_class = get_loss_class(loss)
      self.loss = loss_class()
      if self.loss.recurrent:
        self.recurrent = True
    self.L2 = L2
    self._is_output_layer = is_output_layer

  def __repr__(self):
    return "%s{class=%s, out_type=%s}" % (
      self.name, self.layer_class, self.output.get_description(with_name=False))

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
      session.run(tf.assign(self.params[param_name], values))

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

  def _get_target_value(self):
    """
    :rtype: Data
    """
    target = self.network.extern_data.get_data(self.target)
    return target

  def get_loss_value(self):
    """
    :return: the loss, a scalar value, or None if not set
    :rtype: tf.Tensor | None
    """
    if not self.loss:
      return None
    with tf.name_scope("loss"):
      self.loss.init(
        output=self.output,
        output_before_softmax=self.output_before_softmax,
        target=self._get_target_value())
      return self.loss.get_value()

  def get_error_value(self):
    """
    :return: usually the frame error rate, or None if not defined
    :rtype: tf.Tensor | None
    """
    # TODO ...

  def get_params_l2_norm(self):
    return 2 * sum([tf.nn.l2_loss(param) for (name, param) in sorted(self.params.items())])

  def get_constraints_value(self):
    c = 0
    if self.L2:
      c += self.L2 * self.get_params_l2_norm()
    if c is 0:
      return None
    return c


class SourceLayer(LayerBase):
  layer_class = "source"

  def __init__(self, network, data_key=None, sources=(), **kwargs):
    if data_key is None:
      data_key = network.extern_data.default_input
    assert not sources, "source layer does not expect sources"
    data = network.extern_data.get_data(data_key)
    super(SourceLayer, self).__init__(out_type=data.get_kwargs(), network=network, **kwargs)
    self.output = data


def concat_sources(src_layers):
  """
  :param list[LayerBase] src_layers:
  :return: data with placeholders set
  :rtype: Data
  """
  assert src_layers, "need source layers"
  if len(src_layers) == 1:
    return src_layers[0].output
  assert not src_layers[0].output.sparse, "sparse concat not supported"
  shape = src_layers[0].output.shape
  assert shape, "source must not be a scalar of layer %r" % src_layers[0]
  prefix_shape = shape[:-1]
  dim = 0
  dtype = src_layers[0].output.dtype
  for layer in src_layers:
    assert layer.output.dtype == dtype, "incompatible dtype with layer %r" % layer
    shape = layer.output.shape
    assert shape, "source must not be a scalar of layer %r" % layer
    assert shape[:-1] == prefix_shape, "incompatible concat with layer %r" % layer
    assert shape[-1], "source last-dim must be specified of layer %r" % layer
    dim += shape[-1]
  data = Data(
    name="concat_sources",
    shape=prefix_shape + (dim,),
    dim=dim,
    sparse=False,
    dtype=dtype)
  data.placeholder = tf.concat(
    concat_dim=len(prefix_shape),
    values=[layer.output.placeholder for layer in src_layers])
  data.size_placeholder = src_layers[0].output.size_placeholder
  return data


class _ConcatInputLayer(LayerBase):
  def __init__(self, dropout=0, mask=None, **kwargs):
    super(_ConcatInputLayer, self).__init__(**kwargs)
    input_data = concat_sources(self.sources)

    if self.network.train_flag:
      assert mask in ['dropout', 'unity', None], "invalid mask: %r" % mask
      if mask == "dropout" or (mask is None and dropout > 0):
        assert 0.0 < dropout < 1.0
        input_data.placeholder = tf.nn.dropout(
          input_data.placeholder,
          keep_prob=1 - dropout,
          # noise_shape is like old behavior for now:
          # all dynamic dimensions (batch,time) will use the same dropout-mask broadcasted.
          noise_shape=[1 if dim is None else dim
                       for dim in input_data.batch_shape],
          seed=self.network.random.randint(2**31))

    self.input_data = input_data


class LinearLayer(_ConcatInputLayer):
  layer_class = "linear"

  def __init__(self, activation, with_bias=True, **kwargs):
    super(LinearLayer, self).__init__(**kwargs)

    self.activation = activation
    self.with_bias = with_bias

    input_data = self.input_data
    n_in = input_data.shape[-1]
    n_out = self.output.shape[-1]
    assert n_in and n_out, "%r and %r" % (input_data.shape, self.output.shape)

    W = self.add_param(
      tf.Variable(
        name="W",
        initial_value=tf.contrib.layers.xavier_initializer()(
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
      x = dot(input_data.placeholder, W)

      if self.with_bias:
        x = tf.add(x, b, name="add_bias")

    if self.activation:
      from TFUtil import get_activation_function
      act_func = get_activation_function(self.activation)
      if act_func is tf.nn.softmax:
        self.output_before_softmax = x
      with tf.name_scope("activation"):
        x = act_func(x)

    self.output.placeholder = x


class RecLayer(_ConcatInputLayer):
  layer_class = "rec"
  _rnn_cells_dict = {}

  @classmethod
  def _create_rnn_cells_dict(cls):
    from tensorflow.python.ops import rnn_cell
    for key, v in vars(rnn_cell).items():
      if isinstance(v, type) and issubclass(v, rnn_cell.RNNCell):
        name = key
        if name.endswith("Cell"):
          name = name[:-len("Cell")]
        name = name.lower()
        assert name not in cls._rnn_cells_dict
        cls._rnn_cells_dict[name] = v

  def __init__(self, bidirectional=False, unit="basiclstm", **kwargs):
    super(RecLayer, self).__init__(**kwargs)
    from tensorflow.python.ops import rnn, rnn_cell
    if not self._rnn_cells_dict:
      self._create_rnn_cells_dict()
    rnn_cell_class = self._rnn_cells_dict[unit.lower()]
    with tf.name_scope("rec") as scope:
      n_hidden = self.output.dim
      if bidirectional:
        assert n_hidden % 2 == 0
        n_hidden //= 2
      cell_fw = rnn_cell_class(n_hidden)
      assert isinstance(cell_fw, rnn_cell.RNNCell)
      if bidirectional:
        cell_bw = rnn_cell_class(n_hidden)
      else:
        cell_bw = None
      x = tf.transpose(self.input_data.placeholder, perm=(1, 0, 2))  # (time,batch,dim)
      seq_len = self.input_data.size_placeholder[0]
      if bidirectional:
        # Will get (time,batch,ydim/2).
        (y_fw, y_bw), _ = rnn.bidirectional_dynamic_rnn(
          cell_fw=cell_fw, cell_bw=cell_bw,
          inputs=x, sequence_length=seq_len,
          dtype="float32")
        y = tf.concat(2, (y_fw, y_bw))  # (time,batch,ydim)
      else:
        # Will get (time,batch,ydim).
        y, _ = rnn.dynamic_rnn(cell=cell_fw, inputs=x, sequence_length=seq_len, dtype="float32")
      self.output.placeholder = tf.transpose(y, perm=(1, 0, 2))  # (time,batch,dim)
      params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
      print(params, [p.name[len(scope):-2] for p in params])  # TODO...
      self.params.update({p.name[len(scope):-2]: p for p in params})


class SoftmaxLayer(LinearLayer):
  layer_class = "softmax"

  def __init__(self, activation="softmax", **kwargs):
    super(SoftmaxLayer, self).__init__(activation=activation, **kwargs)


class Loss(object):
  class_name = None
  recurrent = False  # if this is a frame-wise criteria, this will be False

  def __init__(self):
    # All are initialized in self.init().
    self.output = None  # type: Data
    self.output_before_softmax = None  # type: tf.Tensor
    self.output_seq_lens = None  # type: tf.Tensor
    self.target = None  # type: Data
    self.target_seq_lens = None  # type: tf.Tensor
    self.output_flat = None  # type: tf.Tensor
    self.output_before_softmax_flat = None  # type: tf.Tensor
    self.target_flat = None  # type: tf.Tensor
    # Maybe make configurable. For now, same as in our Theano behavior.
    self.reduce_func = tf.reduce_sum  # or tf.reduce_mean

  def init(self, output, output_before_softmax=None, target=None):
    """
    :param Data output: generated output
    :param tf.Tensor|None output_before_softmax: if output is softmax(x), this will be x
    :param Data target: reference target from dataset
    """
    from TFUtil import flatten_with_seq_len_mask
    with tf.name_scope("loss_init"):
      self.output = output
      self.output_before_softmax = output_before_softmax
      self.output_seq_lens = output.size_placeholder[0]
      self.target = target
      self.target_seq_lens = target.size_placeholder[0]
      # Flat variants are with batch,time collapsed into one, masked via seq_lens.
      self.output_flat = None
      self.output_before_softmax_flat = None
      if output_before_softmax is not None:
        self.output_before_softmax_flat = flatten_with_seq_len_mask(output_before_softmax, self.target_seq_lens)
      else:
        self.output_flat = flatten_with_seq_len_mask(output.placeholder, self.target_seq_lens)
      self.target_flat = flatten_with_seq_len_mask(target.placeholder, self.target_seq_lens)

  def get_error(self):
    """
    :return: frame error rate as a scalar value
    :rtype: tf.Tensor
    """
    with tf.name_scope("loss_frame_error"):
      from TFUtil import check_input_ndim, check_shape_equal
      output_flat = check_input_ndim(self.output_flat, ndim=2)
      last_dim = tf.rank(self.output_flat) - 1  # should be 1
      if self.target.sparse:
        target_flat = check_input_ndim(self.target_flat, ndim=1)
        return self.reduce_func(tf.cast(tf.not_equal(
          tf.arg_max(output_flat, dimension=last_dim),
          target_flat
        ), dtype="float32"))
      else:
        target_flat = check_shape_equal(self.target_flat, output_flat)
        return self.reduce_func(tf.cast(tf.not_equal(
          tf.arg_max(output_flat, dimension=last_dim),
          tf.arg_max(target_flat, dimension=last_dim)
        ), dtype="float32"))

  def get_value(self):
    """
    :return: loss as a scalar value
    :rtype: tf.Tensor
    """
    raise NotImplementedError


class CrossEntropyLoss(Loss):
  class_name = "ce"

  def get_value(self):
    with tf.name_scope("loss_ce"):
      if self.target.sparse:
        if self.output_before_softmax_flat is not None:
          out = tf.nn.sparse_softmax_cross_entropy_with_logits(self.output_before_softmax_flat, self.target_flat)
          return self.reduce_func(out)
        else:
          out = tf.log(tf.gather(self.output_flat, self.target_flat))
          return -self.reduce_func(out)
      else:  # not sparse
        if self.output_before_softmax_flat is not None:
          out = tf.nn.softmax_cross_entropy_with_logits(self.output_before_softmax_flat, self.target_flat)
          return self.reduce_func(out)
        else:
          out = self.target_flat * tf.log(self.output_flat)
          return -self.reduce_func(out)


class CtcLoss(Loss):
  class_name = "ctc"
  recurrent = True

  def get_value(self):
    with tf.name_scope("loss_ctc"):
      if self.target.sparse:
        logits = self.output_before_softmax
        if logits is None:
          logits = tf.log(self.output)
        seq_lens = self.output_seq_lens
        from TFUtil import sparse_labels
        labels = sparse_labels(self.target.placeholder, self.target_seq_lens)
        loss = tf.nn.ctc_loss(inputs=logits, labels=labels, sequence_length=seq_lens, time_major=True)
        return tf.reduce_mean(loss)  # or self.reduce_func?
      else:  # not sparse
        raise Exception("CTC target expected to be sparse (symbols)")


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
  :rtype: () -> LayerBase
  """
  if not _LayerClassDict:
    _init_layer_class_dict()
  if name not in _LayerClassDict:
    raise Exception("unknown layer class %r" % name)
  return _LayerClassDict[name]

