
import tensorflow as tf
from TFUtil import Data


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
      alternatively, this also can be a layer name.
    :param str|None loss: if set, via get_loss
    :param float|None L2: for constraints
    :param bool|None is_output_layer:
    """
    self.name = name
    self.network = network
    if loss and not target:
      target = self.network.extern_data.default_target
    self.target = target
    self.loss = None  # type: Loss
    if loss:
      loss_class = get_loss_class(loss)
      self.loss = loss_class()
      if self.loss.recurrent:
        self.recurrent = True
    if out_type is None and n_out is None and target:
      n_out = self._get_target_value(mark_data_key_as_used=False).dim
      if self.loss:
        n_out = self.loss.get_auto_output_layer_dim(n_out)
    if out_type is None:
      assert n_out
      out_type = {"dim": n_out}
    out_type = out_type.copy()
    out_type.setdefault("name", "%s_output" % self.name)
    if n_out is not None:
      out_type.setdefault("dim", n_out)
      assert out_type["dim"] == n_out
    # You are supposed to set self.output.{batch_dim_axis,time_dim_axis} explicitly,
    # as well as check the inputs if they are as you would suggest.
    # However, a good default is often to use the same as the input.
    if sources and "batch_dim_axis" not in out_type:
      out_type.setdefault("batch_dim_axis", sources[0].output.batch_dim_axis)
      out_type.setdefault("time_dim_axis", sources[0].output.time_dim_axis)
    self.output = Data(**out_type)
    # You are not supposed to set self.output.placeholder to the value which you want to return by the layer.
    # Normally you are also supposed to set self.output.size_placeholder explicitly, just like self.output.placeholder.
    # However, in many cases, this will just be {0: time-lengths} and the same as from the input.
    # We check for this case and preset it by that if possible.
    # If you want to have it different in your layer, just overwrite it.
    if sources and sources[0].output.matches_dim_pattern(self.output):
      self.output.size_placeholder = sources[0].output.size_placeholder.copy()
    self.output_before_softmax = None  # type: None|tf.Tensor
    self.sources = sources
    self.params = {}  # type: dict[str,tf.Variable]
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

  def _get_target_value(self, mark_data_key_as_used=True):
    """
    :param bool mark_data_key_as_used: forwarded self.network.get_extern_data()
    :rtype: Data | None
    """
    if not self.target or self.target == "none":
      return None
    if self.network.extern_data.has_data(self.target):
      return self.network.get_extern_data(self.target, mark_data_key_as_used=mark_data_key_as_used)
    if self.target in self.network.layers:
      return self.network.layers[self.target].output
    raise Exception("target %r unknown" % self.target)

  def _init_loss(self):
    if self.loss.output is self.output:
      return
    self.loss.init(
      output=self.output,
      output_before_softmax=self.output_before_softmax,
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
    """
    :param TFNetwork.TFNetwork network:
    :param str|None data_key:
    :param tuple sources:
    """
    if data_key is None:
      data_key = network.extern_data.default_input
    assert not sources, "source layer does not expect sources"
    data = network.get_extern_data(data_key)
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
  shape = src_layers[0].output.shape  # without batch-dim
  assert shape, "source must not be a scalar of layer %r" % src_layers[0]
  prefix_shape = shape[:-1]
  dim = 0
  dtype = src_layers[0].output.dtype
  batch_dim_axis = src_layers[0].output.batch_dim_axis
  time_dim_axis = src_layers[0].output.time_dim_axis
  for layer in src_layers:
    assert layer.output.dtype == dtype, "incompatible dtype with layer %r" % layer
    assert layer.output.batch_dim_axis == batch_dim_axis
    assert layer.output.time_dim_axis == time_dim_axis
    shape = layer.output.shape
    assert layer.output.placeholder.get_shape().ndims == len(shape) + 1  # with batch-dim
    assert shape, "source must not be a scalar of layer %r" % layer
    assert shape[:-1] == prefix_shape, "incompatible concat with layer %r" % layer
    assert shape[-1], "source last-dim must be specified of layer %r" % layer
    dim += shape[-1]
  data = Data(
    name="concat_sources",
    shape=prefix_shape + (dim,),
    dim=dim,
    sparse=False,
    batch_dim_axis=batch_dim_axis,
    time_dim_axis=time_dim_axis,
    dtype=dtype)
  data.placeholder = tf.concat(
    concat_dim=len(prefix_shape) + 1,  # one more because this is with batch-dim
    values=[layer.output.placeholder for layer in src_layers])
  data.size_placeholder = src_layers[0].output.size_placeholder.copy()
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
          noise_shape=input_data.default_broadcast_noise_shape,
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

      x = dot(x, W)
      assert x.get_shape().ndims == ndim

      if self.with_bias:
        x = tf.add(x, b, name="add_bias")
        assert x.get_shape().ndims == ndim

    if self.activation:
      from TFUtil import get_activation_function
      act_func = get_activation_function(self.activation)
      if act_func is tf.nn.softmax:
        self.output_before_softmax = x
      with tf.name_scope("activation"):
        x = act_func(x)

    self.output.batch_dim_axis = self.input_data.batch_dim_axis
    self.output.time_dim_axis = self.input_data.time_dim_axis
    self.output.placeholder = x


class RecLayer(_ConcatInputLayer):
  layer_class = "rec"
  recurrent = True
  _rnn_cells_dict = {}

  @classmethod
  def _create_rnn_cells_dict(cls):
    from tensorflow.python.ops import rnn_cell
    import tensorflow.contrib.rnn as rnn_contrib
    import TFNativeOp
    def maybe_add(key, v):
      if isinstance(v, type) and issubclass(v, (rnn_cell.RNNCell, rnn_contrib.FusedRNNCell, TFNativeOp.RecSeqCellOp)):
        name = key
        if name.endswith("Cell"):
          name = name[:-len("Cell")]
        name = name.lower()
        assert name not in cls._rnn_cells_dict
        cls._rnn_cells_dict[name] = v
    for key, v in vars(rnn_cell).items():
      maybe_add(key, v)
    for key, v in vars(rnn_contrib).items():
      maybe_add(key, v)
    for key, v in vars(TFNativeOp).items():
      maybe_add(key, v)

  def __init__(self, unit="lstm", bidirectional=False, direction=None, **kwargs):
    super(RecLayer, self).__init__(**kwargs)
    from tensorflow.python.ops import rnn, rnn_cell
    import tensorflow.contrib.rnn as rnn_contrib
    import TFNativeOp
    if unit in ["lstmp", "lstm"]:
      # Some possible LSTM implementations are:
      # * BasicLSTM, via official TF, pure TF implementation
      # * LSTMBlockFused, via tf.contrib.rnn (both CPU and GPU). should be much faster than BasicLSTM
      # * NativeLSTM, our own native LSTM (both CPU and GPU). should be faster than LSTMBlockFused
      # We default to the fastest one, i.e. NativeLSTM.
      # Note that they are currently not compatible to each other, i.e. the way the parameters are represented.
      unit = "nativelstm"
    if direction is not None:
      assert not bidirectional
      assert direction in [-1, 1]
    if not self._rnn_cells_dict:
      self._create_rnn_cells_dict()
    rnn_cell_class = self._rnn_cells_dict[unit.lower()]
    with tf.variable_scope(
          "rec",
          initializer=tf.contrib.layers.xavier_initializer(
            seed=self.network.random.randint(2**31))) as scope:
      assert isinstance(scope, tf.VariableScope)
      scope_name_prefix = scope.name + "/"  # e.g. "layer1/rec/"
      n_hidden = self.output.dim
      if bidirectional:
        assert n_hidden % 2 == 0
        n_hidden //= 2
      cell_fw = rnn_cell_class(n_hidden)
      assert isinstance(cell_fw, (rnn_cell.RNNCell, rnn_contrib.FusedRNNCell, TFNativeOp.RecSeqCellOp))  # e.g. BasicLSTMCell
      if bidirectional:
        cell_bw = rnn_cell_class(n_hidden)
      else:
        cell_bw = None
      x = self.input_data.placeholder  # (batch,time,dim) or (time,batch,dim)
      if not self.input_data.is_time_major:
        assert self.input_data.batch_dim_axis == 0
        assert self.input_data.time_dim_axis == 1
        x = tf.transpose(x, [1, 0, 2])   # (time,batch,dim)
      seq_len = self.input_data.size_placeholder[0]
      if isinstance(cell_fw, (rnn_cell.RNNCell, rnn_contrib.FusedRNNCell)):
        if direction == -1:
          x = tf.reverse_sequence(x, seq_lengths=seq_len, batch_dim=1, seq_dim=0)
        if isinstance(cell_fw, rnn_cell.RNNCell):  # e.g. BasicLSTMCell
          if bidirectional:
            # Will get (time,batch,ydim/2).
            (y_fw, y_bw), _ = rnn.bidirectional_dynamic_rnn(
              cell_fw=cell_fw, cell_bw=cell_bw,
              inputs=x, time_major=True, sequence_length=seq_len,
              dtype=tf.float32)
            y = tf.concat(2, (y_fw, y_bw))  # (time,batch,ydim)
          else:
            # Will get (time,batch,ydim).
            y, _ = rnn.dynamic_rnn(cell=cell_fw, inputs=x, time_major=True, sequence_length=seq_len, dtype=tf.float32)
        elif isinstance(cell_fw, rnn_contrib.FusedRNNCell):  # e.g. LSTMBlockFusedCell
          if bidirectional:
            raise NotImplementedError
          # Will get (time,batch,ydim).
          y, _ = cell_fw(inputs=x, sequence_length=seq_len, dtype=tf.float32)
        else:
          raise Exception("invalid type: %s" % type(cell_fw))
        if direction == -1:
          y = tf.reverse_sequence(y, seq_lengths=seq_len, batch_dim=1, seq_dim=0)
      elif isinstance(cell_fw, TFNativeOp.RecSeqCellOp):
        assert not bidirectional
        W = tf.get_variable(name="W", shape=(self.input_data.dim, cell_fw.n_input_dim), dtype=tf.float32)
        b = tf.get_variable(name="b", shape=(cell_fw.n_input_dim,), dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        from TFUtil import dot, sequence_mask_time_major, directed
        x = dot(x, W) + b
        index = sequence_mask_time_major(seq_len, maxlen=tf.shape(x)[0])
        y = cell_fw(inputs=directed(x, direction), index=directed(index, direction))
        y = directed(y, direction)
      else:
        raise Exception("invalid type: %s" % type(cell_fw))
      self.output.time_dim_axis = 0
      self.output.batch_dim_axis = 1
      self.output.placeholder = y
      params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name_prefix)
      assert params
      self.params.update({p.name[len(scope_name_prefix):-2]: p for p in params})


class SoftmaxLayer(LinearLayer):
  layer_class = "softmax"

  def __init__(self, activation="softmax", **kwargs):
    super(SoftmaxLayer, self).__init__(activation=activation, **kwargs)


class FsaLayer(LayerBase):
  layer_class = "fsa"

  def __init__(self, **kwargs):
    """
    """
    super(FsaLayer, self).__init__(**kwargs)
    # TODO...


class Loss(object):
  class_name = None
  recurrent = False  # if this is a frame-wise criteria, this will be False

  def __init__(self):
    # All are initialized in self.init().
    self.output = None  # type: Data
    self.time_major = None  # type: bool|None
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
        self.output_before_softmax_flat = flatten_with_seq_len_mask(output_before_softmax, self.output_seq_lens, time_major=output.is_time_major)
      else:
        self.output_flat = flatten_with_seq_len_mask(output.placeholder, self.output_seq_lens, time_major=output.is_time_major)
      self.target_flat = flatten_with_seq_len_mask(target.placeholder, self.target_seq_lens, time_major=target.is_time_major)

  def get_error(self):
    """
    :return: frame error rate as a scalar value
    :rtype: tf.Tensor
    """
    with tf.name_scope("loss_frame_error"):
      from TFUtil import check_input_ndim, check_shape_equal
      output_flat = self.output_before_softmax_flat
      if output_flat is None:
        output_flat = self.output_flat
      output_flat = check_input_ndim(output_flat, ndim=2)
      last_dim = tf.rank(output_flat) - 1  # should be 1
      output_label = tf.cast(tf.arg_max(output_flat, dimension=last_dim), "int32")
      if self.target.sparse:
        target_label = check_input_ndim(self.target_flat, ndim=1)
      else:
        target_flat = check_shape_equal(self.target_flat, output_flat)
        target_label = tf.cast(tf.arg_max(target_flat, dimension=last_dim), "int32")
      not_equal = tf.not_equal(output_label, target_label)
      return self.reduce_func(tf.cast(not_equal, "float32"))

  def get_value(self):
    """
    :return: loss as a scalar value
    :rtype: tf.Tensor
    """
    raise NotImplementedError

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


class GenericCELoss(Loss):
  class_name = "generic_ce"

  def get_value(self):
    # Should be generic for any activation function.
    # (Except when the labels are not independent, such as for softmax.)
    y = self.p_y_given_x  # Can be anything, e.g. exp or sigmoid, but not softmax.
    y /= T.sum(y, axis=2, keepdims=True)
    nlog_scores = -T.log(T.clip(y, numpy.float32(1.e-20), numpy.float(1.e20)))
    from TheanoUtil import class_idx_seq_to_1_of_k
    y_idx = self.y
    assert y_idx.ndim == 2
    bw = class_idx_seq_to_1_of_k(y_idx, num_classes=self.attrs["n_out"])
    assert bw.ndim == 3
    err_inner = bw * nlog_scores
    src_index = self.sources[0].index
    float_idx = T.cast(src_index, "float32")
    float_idx_bc = float_idx.dimshuffle(0, 1, 'x')
    err = (err_inner * float_idx_bc).sum()
    grad_f = T.grad(None, self.z, known_grads={T.log(self.p_y_given_x): T.ones(y.shape, y.dtype)})
    known_grads = {self.z: grad_f * (y - bw) * float_idx_bc}
    return err, known_grads


class CtcLoss(Loss):
  class_name = "ctc"
  recurrent = True

  def get_value(self):
    if not self.target.sparse:
      raise Exception("CTC target expected to be sparse (symbols)")
    with tf.name_scope("loss_ctc"):
      logits = self.output_before_softmax
      if logits is None:
        logits = tf.log(self.output)
      assert logits.get_shape().ndims == 3  # (B,T,N) or (T,B,N)
      assert logits.get_shape().dims[2].value == self.target.dim + 1  # one more for blank
      seq_lens = self.output_seq_lens
      from TFUtil import sparse_labels
      labels = sparse_labels(self.target.placeholder, self.target_seq_lens)
      loss = tf.nn.ctc_loss(inputs=logits, labels=labels, sequence_length=seq_lens, time_major=self.output.is_time_major)
      return self.reduce_func(loss)

  def get_error(self):
    if not self.target.sparse:
      raise Exception("CTC target expected to be sparse (symbols)")
    with tf.name_scope("loss_ctc_error"):
      logits = self.output_before_softmax
      if logits is None:
        logits = tf.log(self.output)
      if not self.output.is_time_major:
        logits = tf.transpose(logits, [1, 0, 2])  # (B,T,N) => (T,B,N)
      seq_lens = self.output_seq_lens
      decoded, _ = tf.nn.ctc_greedy_decoder(inputs=logits, sequence_length=seq_lens)
      from TFUtil import sparse_labels
      labels = sparse_labels(self.target.placeholder, self.target_seq_lens)
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
  :rtype: () -> LayerBase
  """
  if not _LayerClassDict:
    _init_layer_class_dict()
  if name not in _LayerClassDict:
    raise Exception("unknown layer class %r" % name)
  return _LayerClassDict[name]

