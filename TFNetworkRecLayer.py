
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops.nn import rnn_cell
from TFNetworkLayer import LayerBase, _ConcatInputLayer, SearchChoices, get_concat_sources_data_template
from TFUtil import Data, reuse_name_scope
from Log import log


class RecLayer(_ConcatInputLayer):
  """
  Recurrent layer, has support for several implementations of LSTMs (via ``unit`` argument),
  see :ref:`tf_lstm_benchmark` (http://returnn.readthedocs.io/en/latest/tf_lstm_benchmark.html),
  and also GRU, or simple RNN.

  A subnetwork can also be given which will be evaluated step-by-step,
  which can use attention over some separate input,
  which can be used to implement a decoder in a sequence-to-sequence scenario.
  The subnetwork will get the extern data from the parent net as templates,
  and if there is input to us, then it will be available as the "source" data key.
  """

  layer_class = "rec"
  recurrent = True
  # Some possible LSTM implementations are (in all cases for both CPU and GPU):
  # * BasicLSTM (the cell), via official TF, pure TF implementation
  # * LSTMBlock (the cell), via tf.contrib.rnn.
  # * LSTMBlockFused, via tf.contrib.rnn. should be much faster than BasicLSTM
  # * NativeLSTM, our own native LSTM. should be faster than LSTMBlockFused
  # * CudnnLSTM, via tf.contrib.cudnn_rnn. This is experimental yet.
  # We default to the current tested fastest one, i.e. NativeLSTM.
  # Note that they are currently not compatible to each other, i.e. the way the parameters are represented.
  _default_lstm_unit = "nativelstm"  # TFNativeOp.NativeLstmCell

  def __init__(self,
               unit="lstm", unit_opts=None,
               direction=None, input_projection=True,
               initial_state=None,
               max_seq_len=None,
               forward_weights_init=None, recurrent_weights_init=None, bias_init=None,
               optimize_move_layers_out=None,
               cheating=False,
               **kwargs):
    """
    :param str|dict[str,dict[str]] unit: the RNNCell/etc name, e.g. "nativelstm". see comment below.
      alternatively a whole subnetwork, which will be executed step by step,
      and which can include "prev" in addition to "from" to refer to previous steps.
    :param None|dict[str] unit_opts: passed to RNNCell creation
    :param int|None direction: None|1 -> forward, -1 -> backward
    :param bool input_projection: True -> input is multiplied with matrix. False only works if same input dim
    :param LayerBase|str|float|int|tuple|None initial_state:
    :param int|tf.Tensor|None max_seq_len: if unit is a subnetwork. str will be evaluated. see code
    :param str forward_weights_init: see :func:`TFUtil.get_initializer`
    :param str recurrent_weights_init: see :func:`TFUtil.get_initializer`
    :param str bias_init: see :func:`TFUtil.get_initializer`
    :param bool|None optimize_move_layers_out: will automatically move layers out of the loop when possible
    :param bool cheating: make targets available, and determine length by them
    """
    super(RecLayer, self).__init__(**kwargs)
    import re
    from TFUtil import is_gpu_available
    from tensorflow.contrib import rnn as rnn_contrib
    from tensorflow.python.util import nest
    if is_gpu_available():
      from tensorflow.contrib import cudnn_rnn
    else:
      cudnn_rnn = None
    import TFNativeOp
    if direction is not None:
      assert direction in [-1, 1]
    self._last_hidden_state = None
    self._direction = direction
    self._initial_state_deps = [l for l in nest.flatten(initial_state) if isinstance(l, LayerBase)]
    self._input_projection = input_projection
    self._max_seq_len = max_seq_len
    if optimize_move_layers_out is None:
      optimize_move_layers_out = self.network.get_config().bool("optimize_move_layers_out", True)
    self._optimize_move_layers_out = optimize_move_layers_out
    self._cheating = cheating
    self._sub_loss = None
    self._sub_error = None
    self._sub_loss_normalization_factor = None
    # On the random initialization:
    # For many cells, e.g. NativeLSTM: there will be a single recurrent weight matrix, (output.dim, output.dim * 4),
    # and a single input weight matrix (input_data.dim, output.dim * 4), and a single bias (output.dim * 4,).
    # The bias is by default initialized with 0.
    # In the Theano :class:`RecurrentUnitLayer`, create_recurrent_weights() and create_forward_weights() are used,
    #   where forward_weights_init = "random_uniform(p_add=%i)" % (output.dim * 4)
    #   and recurrent_weights_init = "random_uniform()",
    #   thus with in=input_data.dim, out=output.dim,
    #   for forward weights: uniform sqrt(6. / (in + out*8)), for rec. weights: uniform sqrt(6. / (out*5)).
    # TensorFlow initializers:
    #   https://www.tensorflow.org/api_guides/python/contrib.layers#Initializers
    #   https://www.tensorflow.org/api_docs/python/tf/orthogonal_initializer
    #   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/init_ops.py
    #   xavier_initializer with uniform=True: uniform sqrt(6 / (fan_in + fan_out)),
    #     i.e. uniform sqrt(6. / (in + out*4)) for forward, sqrt(6./(out*5)) for rec.
    #     Ref: https://www.tensorflow.org/api_docs/python/tf/contrib/layers/xavier_initializer
    # Keras uses these defaults:
    #   Ref: https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py
    #   Ref: https://keras.io/initializers/, https://github.com/fchollet/keras/blob/master/keras/engine/topology.py
    #   (fwd weights) kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    #   where glorot_uniform is sqrt(6 / (fan_in + fan_out)), i.e. fwd weights: uniform sqrt(6 / (in + out*4)),
    #   and orthogonal creates a random orthogonal matrix (fan_in, fan_out), i.e. rec (out, out*4).
    self._bias_initializer = tf.constant_initializer(0.0)
    self._fwd_weights_initializer = None
    self._rec_weights_initializer = None
    from TFUtil import get_initializer, xavier_initializer
    if forward_weights_init is not None:
      self._fwd_weights_initializer = get_initializer(
        forward_weights_init, seed=self.network.random.randint(2**31), eval_local_ns={"layer": self})
    if recurrent_weights_init is not None:
      self._rec_weights_initializer = get_initializer(
        recurrent_weights_init, seed=self.network.random.randint(2**31), eval_local_ns={"layer": self})
    if bias_init is not None:
      self._bias_initializer = get_initializer(
        bias_init, seed=self.network.random.randint(2**31), eval_local_ns={"layer": self})
    if self._rec_weights_initializer:
      default_var_initializer = self._rec_weights_initializer
    elif self._fwd_weights_initializer:
      default_var_initializer = self._fwd_weights_initializer
    else:
      default_var_initializer = xavier_initializer(seed=self.network.random.randint(2**31))
    with reuse_name_scope("rec", initializer=default_var_initializer) as scope:
      assert isinstance(scope, tf.VariableScope)
      self._rec_scope = scope
      scope_name_prefix = scope.name + "/"  # e.g. "layer1/rec/"
      with self.var_creation_scope():
        self._initial_state = RnnCellLayer.get_rec_initial_state(
          initial_state=initial_state, n_out=self.output.dim, unit=unit, unit_opts=unit_opts,
          batch_dim=self.network.get_data_batch_dim(), name=self.name,
          rec_layer=self) if initial_state is not None else None
        self.cell = self._get_cell(unit, unit_opts=unit_opts)
        if isinstance(self.cell, (rnn_cell.RNNCell, rnn_contrib.FusedRNNCell)):
          y = self._get_output_cell(self.cell)
        elif cudnn_rnn and isinstance(self.cell, (cudnn_rnn.CudnnLSTM, cudnn_rnn.CudnnGRU)):
          y = self._get_output_cudnn(self.cell)
        elif isinstance(self.cell, TFNativeOp.RecSeqCellOp):
          y = self._get_output_native_rec_op(self.cell)
        elif isinstance(self.cell, _SubnetworkRecCell):
          y = self._get_output_subnet_unit(self.cell)
        else:
          raise Exception("invalid type: %s" % type(self.cell))
        self.output.time_dim_axis = 0
        self.output.batch_dim_axis = 1
        self.output.placeholder = y
        # Very generic way to collect all created params.
        # Note that for the TF RNN cells, there is no other way to do this.
        # Also, see the usage of :func:`LayerBase.cls_layer_scope`, e.g. for initial vars.
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=re.escape(scope_name_prefix))
        self._add_params(params=params, scope_name_prefix=scope_name_prefix)
        # More specific way. Should not really add anything anymore but you never know.
        # Also, this will update self.saveable_param_replace.
        if isinstance(self.cell, _SubnetworkRecCell):
          self._add_params(params=self.cell.net.get_params_list(), scope_name_prefix=scope_name_prefix)
          self.saveable_param_replace.update(self.cell.net.get_saveable_param_replace_dict())
          if self.cell.input_layers_net:
            self._add_params(params=self.cell.input_layers_net.get_params_list(), scope_name_prefix=scope_name_prefix)
            self.saveable_param_replace.update(self.cell.input_layers_net.get_saveable_param_replace_dict())
          if self.cell.output_layers_net:
            self._add_params(params=self.cell.output_layers_net.get_params_list(), scope_name_prefix=scope_name_prefix)
            self.saveable_param_replace.update(self.cell.output_layers_net.get_saveable_param_replace_dict())

  def _add_params(self, scope_name_prefix, params):
    """
    :param str scope_name_prefix:
    :param list[tf.Variable] params:
    """
    for p in params:
      if not p.name.startswith(scope_name_prefix):
        continue
      assert p.name.startswith(scope_name_prefix) and p.name.endswith(":0")
      self.params[p.name[len(scope_name_prefix):-2]] = p

  def get_dep_layers(self):
    l = super(RecLayer, self).get_dep_layers()
    l += self._initial_state_deps
    if isinstance(self.cell, _SubnetworkRecCell):
      l += self.cell.get_parent_deps()
    return l

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param TFNetwork.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    if isinstance(d.get("unit"), dict):
      d["n_out"] = d.get("n_out", None)  # disable automatic guessing
    super(RecLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if "initial_state" in d:
      d["initial_state"] = RnnCellLayer.transform_initial_state(
        d["initial_state"], network=network, get_layer=get_layer)
    if isinstance(d.get("unit"), dict):
      def sub_get_layer(name):
        # Only used to resolve deps to base network.
        if name.startswith("base:"):
          return get_layer(name[len("base:"):])
      from TFNetwork import TFNetwork, ExternData
      subnet = TFNetwork(parent_net=network, extern_data=network.extern_data)  # dummy subnet
      for sub in d["unit"].values():  # iterate over the layers of the subnet
        assert isinstance(sub, dict)
        if "class" in sub:
          from TFNetworkLayer import get_layer_class
          class_name = sub["class"]
          cl = get_layer_class(class_name)
          # Operate on a copy because we will transform the dict later.
          # We only need this to resolve any other layer dependencies in the main network.
          cl.transform_config_dict(sub.copy(), network=subnet, get_layer=sub_get_layer)
    if isinstance(d.get("max_seq_len"), str):
      from TFNetwork import LayerNotFound

      def max_len_from(src):
        """
        :param str src: layer name
        :return: max seq-len of the layer output
        :rtype: tf.Tensor
        """
        layer = None
        if src.startswith("base:"):
          # For legacy reasons, this was interpret to be in the subnet, so this should access the current net.
          # However, now we want that this behaves more standard, such that "base:" accesses the parent net,
          # but we also want to not break old configs.
          # We first check whether there is such a layer in the parent net.
          try:
            layer = get_layer(src)
          except LayerNotFound:
            src = src[len("base:"):]  # This will fall-back to the old behavior.
        if not layer:
          layer = get_layer(src)
        return tf.reduce_max(layer.output.get_sequence_lengths(), name="max_seq_len_%s" % layer.tf_scope_name)

      with tf.name_scope("user_max_seq_len"):
        d["max_seq_len"] = eval(d["max_seq_len"], {"max_len_from": max_len_from, "tf": tf})

  @classmethod
  def get_out_data_from_opts(cls, unit, sources=(), initial_state=None, **kwargs):
    from tensorflow.python.util import nest
    n_out = kwargs.get("n_out", None)
    out_type = kwargs.get("out_type", None)
    loss = kwargs.get("loss", None)
    deps = list(sources)  # type: list[LayerBase]
    deps += [l for l in nest.flatten(initial_state) if isinstance(l, LayerBase)]
    if out_type or n_out or loss:
      if out_type:
        assert out_type.get("time_dim_axis", 0) == 0
        assert out_type.get("batch_dim_axis", 1) == 1
      out = super(RecLayer, cls).get_out_data_from_opts(sources=sources, **kwargs)
    else:
      out = None
    if isinstance(unit, dict):  # subnetwork
      source_data = get_concat_sources_data_template(sources) if sources else None
      subnet = _SubnetworkRecCell(parent_net=kwargs["network"], net_dict=unit, source_data=source_data)
      sub_out = subnet.layer_data_templates["output"].output.copy_template_adding_time_dim(
        name="%s_output" % kwargs["name"], time_dim_axis=0)
      if out:
        assert sub_out.dim == out.dim
        assert sub_out.shape == out.shape
      out = sub_out
      deps += subnet.get_parent_deps()
    assert out
    out.time_dim_axis = 0
    out.batch_dim_axis = 1
    cls._post_init_output(output=out, sources=sources, **kwargs)
    for dep in deps:
      out.beam_size = out.beam_size or dep.output.beam_size
    return out

  def get_absolute_name_scope_prefix(self):
    return self.get_base_absolute_name_scope_prefix() + "rec/"  # all under "rec" sub-name-scope

  _rnn_cells_dict = {}

  @classmethod
  def _create_rnn_cells_dict(cls):
    from TFUtil import is_gpu_available
    from tensorflow.contrib import rnn as rnn_contrib
    import TFNativeOp
    allowed_types = (rnn_cell.RNNCell, rnn_contrib.FusedRNNCell, TFNativeOp.RecSeqCellOp)
    if is_gpu_available():
      from tensorflow.contrib import cudnn_rnn
      allowed_types += (cudnn_rnn.CudnnLSTM, cudnn_rnn.CudnnGRU)
    else:
      cudnn_rnn = None
    def maybe_add(key, v):
      if isinstance(v, type) and issubclass(v, allowed_types):
        name = key
        if name.endswith("Cell"):
          name = name[:-len("Cell")]
        name = name.lower()
        assert cls._rnn_cells_dict.get(name) in [v, None]
        cls._rnn_cells_dict[name] = v
    for key, v in globals().items():
      maybe_add(key, v)
    for key, v in vars(rnn_contrib).items():
      maybe_add(key, v)
    for key, v in vars(TFNativeOp).items():
      maybe_add(key, v)
    if is_gpu_available():
      for key, v in vars(cudnn_rnn).items():
        maybe_add(key, v)
    # Alias for the standard LSTM cell, because self._get_cell(unit="lstm") will use "NativeLSTM" by default.
    maybe_add("StandardLSTM", rnn_contrib.LSTMCell)

  _warn_msg_once_for_cell_name = set()

  @classmethod
  def get_rnn_cell_class(cls, name):
    """
    :param str name: cell name, minus the "Cell" at the end
    :rtype: () -> rnn_cell.RNNCell|TFNativeOp.RecSeqCellOp
    """
    if not cls._rnn_cells_dict:
      cls._create_rnn_cells_dict()
    from TFUtil import is_gpu_available
    if not is_gpu_available():
      m = {"cudnnlstm": "LSTMBlockFused", "cudnngru": "GRUBlock"}
      if name.lower() in m:
        if name.lower() not in cls._warn_msg_once_for_cell_name:
          print("You have selected unit %r in a rec layer which is for GPU only, so we are using %r instead." %
                (name, m[name.lower()]), file=log.v2)
          cls._warn_msg_once_for_cell_name.add(name.lower())
        name = m[name.lower()]
    if name.lower() in ["lstmp", "lstm"]:
      name = cls._default_lstm_unit
    return cls._rnn_cells_dict[name.lower()]

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
      x = self.input_data.get_placeholder_as_time_major()  # (time,batch,[dim])
    seq_len = self.input_data.get_sequence_lengths()
    return x, seq_len

  def get_loss_value(self):
    v = super(RecLayer, self).get_loss_value()
    from TFUtil import optional_add
    return optional_add(v, self._sub_loss)

  def get_error_value(self):
    v = super(RecLayer, self).get_error_value()
    if v is not None:
      return v
    return self._sub_error

  def get_loss_normalization_factor(self):
    v = super(RecLayer, self).get_loss_normalization_factor()
    if v is not None:
      return v
    return self._sub_loss_normalization_factor

  def get_constraints_value(self):
    v = super(RecLayer, self).get_constraints_value()
    from TFUtil import optional_add
    if isinstance(self.cell, _SubnetworkRecCell):
      for layer in self.cell.net.layers.values():
        v = optional_add(v, layer.get_constraints_value())
    return v

  def _get_cell(self, unit, unit_opts=None):
    """
    :param str|dict[str] unit:
    :param None|dict[str] unit_opts:
    :rtype: _SubnetworkRecCell|tensorflow.contrib.rnn.RNNCell|tensorflow.contrib.rnn.FusedRNNCell|TFNativeOp.RecSeqCellOp
    """
    from TFUtil import is_gpu_available
    from tensorflow.contrib import rnn as rnn_contrib
    import TFNativeOp
    if isinstance(unit, dict):
      assert unit_opts is None
      return _SubnetworkRecCell(parent_rec_layer=self, net_dict=unit)
    assert isinstance(unit, str)
    rnn_cell_class = self.get_rnn_cell_class(unit)
    n_hidden = self.output.dim
    if unit_opts is None:
      unit_opts = {}
    if is_gpu_available():
      from tensorflow.contrib import cudnn_rnn
      if issubclass(rnn_cell_class, (cudnn_rnn.CudnnLSTM, cudnn_rnn.CudnnGRU)):
        cell = rnn_cell_class(
          num_layers=1, num_units=n_hidden, input_size=self.input_data.dim,
          input_mode='linear_input', direction='unidirectional', dropout=0.0, **unit_opts)
        return cell
    if issubclass(rnn_cell_class, TFNativeOp.RecSeqCellOp):
      cell = rnn_cell_class(
        n_hidden=n_hidden, n_input_dim=self.input_data.dim,
        input_is_sparse=self.input_data.sparse,
        step=self._direction, **unit_opts)
      return cell
    cell = rnn_cell_class(n_hidden, **unit_opts)
    assert isinstance(
      cell, (rnn_contrib.RNNCell, rnn_contrib.FusedRNNCell))  # e.g. BasicLSTMCell
    return cell

  def _get_output_cell(self, cell):
    """
    :param tensorflow.contrib.rnn.RNNCell|tensorflow.contrib.rnn.FusedRNNCell cell:
    :return: output of shape (time, batch, dim)
    :rtype: tf.Tensor
    """
    from tensorflow.python.ops import rnn
    from tensorflow.contrib import rnn as rnn_contrib
    assert self._max_seq_len is None
    assert self.input_data
    assert not self.input_data.sparse
    x, seq_len = self._get_input()
    if isinstance(cell, BaseRNNCell):
      with tf.variable_scope(tf.get_variable_scope(), initializer=self._fwd_weights_initializer):
        x = cell.get_input_transformed(x)
    if self._direction == -1:
      x = tf.reverse_sequence(x, seq_lengths=seq_len, batch_dim=1, seq_dim=0)
    if isinstance(cell, rnn_cell.RNNCell):  # e.g. BasicLSTMCell
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

  @staticmethod
  def _get_cudnn_param_size(num_units, input_size,
                            num_layers=1, rnn_mode="lstm", input_mode="linear_input", direction='unidirectional'):
    """
    :param int num_layers:
    :param int num_units:
    :param int input_size:
    :param str rnn_mode: 'lstm', 'gru', 'rnn_tanh' or 'rnn_relu'
    :param str input_mode: "linear_input", "skip_input", "auto_select". note that we have a different default.
    :param str direction: 'unidirectional' or 'bidirectional'
    :return: size
    :rtype: int
    """
    # Also see test_RecLayer_get_cudnn_params_size().
    dir_count = {"unidirectional": 1, "bidirectional": 2}[direction]
    num_gates = {"lstm": 3, "gru": 2}.get(rnn_mode, 0)
    if input_mode == "linear_input" or (input_mode == "auto_select" and num_units != input_size):
      # (input + recurrent + 2 * bias) * output * (gates + cell in)
      size = (input_size + num_units + 2) * num_units * (num_gates + 1) * dir_count
    elif input_mode == "skip_input" or (input_mode == "auto_select" and num_units == input_size):
      # (recurrent + 2 * bias) * output * (gates + cell in)
      size = (num_units + 2) * num_units * (num_gates + 1) * dir_count
    else:
      raise Exception("invalid input_mode %r" % input_mode)
    # Remaining layers:
    size += (num_units * dir_count + num_units + 2) * num_units * (num_gates + 1) * dir_count * (num_layers - 1)
    return size

  @staticmethod
  def convert_cudnn_canonical_to_lstm_block(reader, prefix, target="lstm_block_wrapper/"):
    """
    This assumes CudnnLSTM currently, with num_layers=1, input_mode="linear_input", direction='unidirectional'!

    :param tf.train.CheckpointReader reader:
    :param str prefix: e.g. "layer2/rec/"
    :param str target: e.g. "lstm_block_wrapper/" or "rnn/lstm_cell/"
    :return: dict key -> value, {".../kernel": ..., ".../bias": ...} with prefix
    :rtype: dict[str,numpy.ndarray]
    """
    # For reference:
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/cudnn_rnn/python/ops/cudnn_rnn_ops.py
    # For CudnnLSTM, there are 8 tensors per weight and per bias for each
    # layer: tensor 0-3 are applied to the input from the previous layer and
    # tensor 4-7 to the recurrent input. Tensor 0 and 4 are for the input gate;
    # tensor 1 and 5 the forget gate; tensor 2 and 6 the new memory gate;
    # tensor 3 and 7 the output gate.
    import numpy
    num_vars = 16
    values = []
    for i in range(num_vars):
      values.append(reader.get_tensor("%scudnn/CudnnRNNParamsToCanonical:%i" % (prefix, i)))
    assert len(values[-1].shape) == 1
    output_dim = values[-1].shape[0]
    # For some reason, the input weight matrices are sometimes flattened.
    assert numpy.prod(values[0].shape) % output_dim == 0
    input_dim = numpy.prod(values[0].shape) // output_dim
    weights_and_biases = [
      (numpy.concatenate(
        [numpy.reshape(values[i], [output_dim, input_dim]),  # input weights
         numpy.reshape(values[i + 4], [output_dim, output_dim])],  # recurrent weights
        axis=1),
       values[8 + i] +  # input bias
       values[8 + i + 4]  # recurrent bias
      )
      for i in range(4)]
    # cuDNN weights are in ifco order, convert to icfo order.
    weights_and_biases[1:3] = reversed(weights_and_biases[1:3])
    weights = numpy.transpose(numpy.concatenate([wb[0] for wb in weights_and_biases], axis=0))
    biases = numpy.concatenate([wb[1] for wb in weights_and_biases], axis=0)
    return {prefix + target + "kernel": weights, prefix + target + "bias": biases}

  def _get_output_cudnn(self, cell):
    """
    :param tensorflow.contrib.cudnn_rnn.CudnnLSTM|tensorflow.contrib.cudnn_rnn.CudnnGRU cell:
    :return: output of shape (time, batch, dim)
    :rtype: tf.Tensor
    """
    from TFUtil import get_current_var_scope_name
    from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
    assert self._max_seq_len is None
    assert self.input_data
    assert not self.input_data.sparse
    x, seq_len = self._get_input()
    n_batch = tf.shape(seq_len)[0]
    if self._direction == -1:
      x = tf.reverse_sequence(x, seq_lengths=seq_len, batch_dim=1, seq_dim=0)
    with tf.variable_scope("cudnn"):
      num_layers = 1
      param_size = self._get_cudnn_param_size(
        num_units=self.output.dim, input_size=self.input_data.dim, rnn_mode=cell._rnn_mode, num_layers=num_layers)
      # Note: The raw params used during training for the cuDNN op is just a single variable
      # with all params concatenated together.
      # For the checkpoint save/restore, we will use Cudnn*Saveable, which also makes it easier in CPU mode
      # to import the params for another unit like LSTMBlockCell.
      # Also see: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/cudnn_rnn/python/kernel_tests/cudnn_rnn_ops_test.py
      params = tf.Variable(
        tf.random_uniform([param_size], minval=-0.01, maxval=0.01, seed=42), name="params_raw", trainable=True)
      if cell._rnn_mode == cudnn_rnn_ops.CUDNN_LSTM:
        fn = cudnn_rnn_ops.CudnnLSTMSaveable
      elif cell._rnn_mode == cudnn_rnn_ops.CUDNN_GRU:
        fn = cudnn_rnn_ops.CudnnGRUSaveable
      elif cell._rnn_mode == cudnn_rnn_ops.CUDNN_RNN_TANH:
        fn = cudnn_rnn_ops.CudnnRNNTanhSaveable
      elif cell._rnn_mode == cudnn_rnn_ops.CUDNN_RNN_RELU:
        fn = cudnn_rnn_ops.CudnnRNNReluSaveable
      else:
        raise ValueError("rnn mode %r" % cell._rnn_mode)
      params_saveable = fn(
        params,
        num_layers=cell.num_layers,
        num_units=cell.num_units,
        input_size=cell.input_size,
        input_mode=cell.input_mode,
        direction=cell.direction,
        scope="%s/params_canonical" % get_current_var_scope_name(),
        name="%s/params_canonical" % get_current_var_scope_name())
      tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
      self.saveable_param_replace[params] = params_saveable
      # It's like a fused cell, i.e. operates on the full sequence.
      input_h = tf.zeros((num_layers, n_batch, self.output.dim), dtype=tf.float32)
      input_c = tf.zeros((num_layers, n_batch, self.output.dim), dtype=tf.float32)
      y, _, _ = cell(x, input_h=input_h, input_c=input_c, params=params)
    if self._direction == -1:
      y = tf.reverse_sequence(y, seq_lengths=seq_len, batch_dim=1, seq_dim=0)
    return y

  def _get_output_native_rec_op(self, cell):
    """
    :param TFNativeOp.RecSeqCellOp cell:
    :return: output of shape (time, batch, dim)
    :rtype: tf.Tensor
    """
    from TFUtil import dot, sequence_mask_time_major, directed, to_int32_64
    assert self._max_seq_len is None
    assert self.input_data
    x, seq_len = self._get_input()
    if self._input_projection:
      if cell.does_input_projection:
        # The cell get's x as-is. It will internally does the matrix mult and add the bias.
        pass
      else:
        W = tf.get_variable(
          name="W", shape=(self.input_data.dim, cell.n_input_dim), dtype=tf.float32,
          initializer=self._fwd_weights_initializer)
        if self.input_data.sparse:
          x = tf.nn.embedding_lookup(W, to_int32_64(x))
        else:
          x = dot(x, W)
        b = tf.get_variable(name="b", shape=(cell.n_input_dim,), dtype=tf.float32, initializer=self._bias_initializer)
        x += b
    else:
      assert not cell.does_input_projection
      assert not self.input_data.sparse
      assert self.input_data.dim == cell.n_input_dim
    index = sequence_mask_time_major(seq_len, maxlen=self.input_data.time_dimension())
    if not cell.does_direction_handling:
      x = directed(x, self._direction)
      index = directed(index, self._direction)
    y, final_state = cell(
      inputs=x, index=index,
      initial_state=self._initial_state,
      recurrent_weights_initializer=self._rec_weights_initializer)
    self._last_hidden_state = final_state
    if not cell.does_direction_handling:
      y = directed(y, self._direction)
    return y

  def _get_output_subnet_unit(self, cell):
    """
    :param _SubnetworkRecCell cell:
    :return: output of shape (time, batch, dim)
    :rtype: tf.Tensor
    """
    output, (sub_loss, sub_error, sub_loss_norm_factor), search_choices = cell.get_output(rec_layer=self)
    self._sub_loss = sub_loss
    self._sub_error = sub_error
    self._sub_loss_normalization_factor = sub_loss_norm_factor
    self.search_choices = search_choices
    return output

  def get_last_hidden_state(self, key):
    assert self._last_hidden_state is not None, (
      "last-hidden-state not implemented/supported for this layer-type. try another unit. see the code.")
    return RnnCellLayer.get_state_by_key(self._last_hidden_state, key=key)

  @classmethod
  def is_prev_step_layer(cls, layer):
    """
    :param LayerBase layer:
    :rtype: bool
    """
    if isinstance(layer, _TemplateLayer):
      return layer.is_prev_time_frame
    return False


class _SubnetworkRecCell(object):
  """
  This class is used by :class:`RecLayer` to implement
  the generic subnetwork logic inside the recurrency.
  """

  _debug_out = None  # set to list to enable

  def __init__(self, net_dict, parent_rec_layer=None, parent_net=None, source_data=None):
    """
    :param dict[str,dict[str]] net_dict: dict for the subnetwork, layer name -> layer dict
    :param RecLayer parent_rec_layer:
    :param TFNetwork.TFNetwork parent_net:
    :param Data|None source_data: usually concatenated input from the rec-layer
    """
    from copy import deepcopy
    if parent_net is None and parent_rec_layer:
      parent_net = parent_rec_layer.network
    if source_data is None and parent_rec_layer:
      source_data = parent_rec_layer.input_data
    self.parent_rec_layer = parent_rec_layer
    self.parent_net = parent_net
    self.net_dict = deepcopy(net_dict)
    from TFNetwork import TFNetwork, ExternData
    self.net = TFNetwork(
      name="%s/%s:rec-subnet" % (parent_net.name, parent_rec_layer.name if parent_rec_layer else "?"),
      extern_data=ExternData(),
      train_flag=parent_net.train_flag,
      search_flag=parent_net.search_flag,
      parent_layer=parent_rec_layer,
      parent_net=parent_net)
    if source_data:
      self.net.extern_data.data["source"] = \
        source_data.copy_template_excluding_time_dim()
    for key in parent_net.extern_data.data.keys():
      if key in self.net.extern_data.data:
        continue  # Don't overwrite existing, e.g. "source".
      # These are just templates. You can use them as possible targets for dimension information,
      # but not as actual sources or targets.
      self.net.extern_data.data[key] = \
        parent_net.extern_data.data[key].copy_template_excluding_time_dim()
    if parent_net.search_flag and parent_rec_layer and parent_rec_layer.output.beam_size:
      for key, data in list(self.net.extern_data.data.items()):
        self.net.extern_data.data[key] = data.copy_extend_with_beam(
          beam_size=parent_rec_layer.output.beam_size)
    self.layer_data_templates = {}  # type: dict[str,_TemplateLayer]
    self.prev_layers_needed = set()  # type: set[str]
    self._construct_template()
    self._initial_outputs = None  # type: dict[str,tf.Tensor]
    self._initial_extra_outputs = None  # type: dict[str,dict[str,tf.Tensor|tuple[tf.Tensor]]]
    self.input_layers_moved_out = []  # type: list[str]
    self.output_layers_moved_out = []  # type: list[str]
    self.layers_in_loop = None   # type: list[str]
    self.input_layers_net = None  # type: TFNetwork
    self.output_layers_net = None  # type: TFNetwork
    self.final_acc_tas_dict = None  # type: dict[str, tf.TensorArray]

  def _construct_template(self):
    """
    Without creating any computation graph, create TemplateLayer instances.
    Need it for shape/meta information as well as dependency graph in advance.
    It will init self.layer_data_templates and self.prev_layers_needed.
    """
    def add_templated_layer(name, layer_class, **layer_desc):
      """
      This is used instead of self.net.add_layer because we don't want to add
      the layers at this point, we just want to construct the template layers
      and store inside self.layer_data_templates.

      :param str name:
      :param type[LayerBase]|LayerBase layer_class:
      :param dict[str] layer_desc:
      :rtype: LayerBase
      """
      # _TemplateLayer already created in get_templated_layer.
      layer = self.layer_data_templates[name]
      layer_desc = layer_desc.copy()
      layer_desc["name"] = name
      layer_desc["network"] = self.net
      output = layer_class.get_out_data_from_opts(**layer_desc)
      layer.init(layer_class=layer_class, output=output, **layer_desc)
      return layer

    class construct_ctx:
      # Stack of layers:
      layers = []  # type: list[_TemplateLayer]

    def get_none_layer(name):
      return None

    def get_templated_layer(name):
      """
      :param str name:
      :rtype: _TemplateLayer|LayerBase
      """
      if name.startswith("prev:"):
        name = name[len("prev:"):]
        self.prev_layers_needed.add(name)
      if name in self.layer_data_templates:
        layer = self.layer_data_templates[name]
        if construct_ctx.layers:
          construct_ctx.layers[-1].dependencies.add(layer)
        return layer
      if name.startswith("base:"):
        layer = self.parent_net.get_layer(name[len("base:"):])
        if construct_ctx.layers:
          construct_ctx.layers[-1].dependencies.add(layer)
        return layer
      # Need to create layer instance here now to not run into recursive loops.
      # We will extend it later in add_templated_layer().
      layer = _TemplateLayer(name=name, network=self.net)
      if construct_ctx.layers:
        construct_ctx.layers[-1].dependencies.add(layer)
      construct_ctx.layers.append(layer)
      self.layer_data_templates[name] = layer
      try:
        # First, see how far we can get without recursive layer construction.
        # We only want to get the data template for now.
        self.net.construct_layer(
          net_dict=self.net_dict, name=name, get_layer=get_none_layer, add_layer=add_templated_layer)
      except Exception:
        # Pretty generic exception handling but anything could happen.
        # Not so nice but should work for now.
        pass
      # Now, do again, but with recursive layer construction, to determine the dependencies.
      self.net.construct_layer(
        net_dict=self.net_dict, name=name, get_layer=get_templated_layer, add_layer=add_templated_layer)
      assert construct_ctx.layers[-1] is layer
      construct_ctx.layers.pop(-1)
      return layer

    assert not self.layer_data_templates, "do not call this multiple times"
    get_templated_layer("output")
    assert "output" in self.layer_data_templates
    assert not construct_ctx.layers

    if "end" in self.net_dict:  # used to specify ending of a sequence
      get_templated_layer("end")

    if self.parent_net.eval_flag:  # only collect losses if we need them
      for layer_name, layer in self.net_dict.items():
        if layer.get("loss"):
          get_templated_layer(layer_name)

  def _construct(self, prev_outputs, prev_extra, i,
                 data=None, classes=None,
                 inputs_moved_out_tas=None, needed_outputs=("output",)):
    """
    :param dict[str,tf.Tensor] prev_outputs: outputs of the layers from the previous step
    :param dict[str,dict[str,tf.Tensor]] prev_extra: extra output / hidden states of the previous step for layers
    :param tf.Tensor i: loop counter. scalar, int32, current step (time)
    :param tf.Tensor|None data: optional source data, shape e.g. (batch,dim)
    :param tf.Tensor|None classes: optional target classes, shape e.g. (batch,) if it is sparse
    :param dict[str,tf.TensorArray]|None inputs_moved_out_tas:
    :param set[str] needed_outputs: layers where we need outputs
    """
    from TFNetwork import TFNetwork
    from TFNetworkLayer import InternalLayer
    from TFUtil import tile_transposed
    needed_beam_size = self.layer_data_templates["output"].output.beam_size
    if data is not None:
      if needed_beam_size:
        assert not self.parent_rec_layer.input_data.beam_size
        data = tile_transposed(
          data,
          axis=self.net.extern_data.data["source"].batch_dim_axis,
          multiples=needed_beam_size)
      self.net.extern_data.data["source"].placeholder = data
    if classes is not None:
      if needed_beam_size:
        classes = tile_transposed(
          classes,
          axis=self.net.extern_data.data[self.parent_rec_layer.target].batch_dim_axis,
          multiples=needed_beam_size)
      self.net.extern_data.data[self.parent_rec_layer.target].placeholder = classes
    for data_key, data in self.net.extern_data.data.items():
      if data_key not in self.net.used_data_keys:
        continue
      if data.placeholder is None:
        raise Exception("rec layer %r subnet data key %r is not set" % (self.parent_rec_layer.name, data_key))

    prev_layers = {}  # type: dict[str,_TemplateLayer]
    for name in set(list(prev_outputs.keys()) + list(prev_extra.keys())):
      self.net.layers["prev:%s" % name] = prev_layers[name] = self.layer_data_templates[name].copy_as_prev_time_frame(
        prev_output=prev_outputs.get(name, None),
        rec_vars_prev_outputs=prev_extra.get(name, None))
    extended_layers = {}

    from copy import deepcopy
    net_dict = deepcopy(self.net_dict)
    for name in net_dict.keys():
      if name in prev_layers:
        net_dict[name]["rec_previous_layer"] = prev_layers[name]

    inputs_moved_out = {}  # type: dict[str,InternalLayer]

    def get_input_moved_out(name):
      """
      :param str name:
      :rtype: InternalLayer
      """
      if name in inputs_moved_out:
        return inputs_moved_out[name]
      if name.startswith("prev:"):
        layer_name = name[len("prev:"):]
        prev = True
        assert layer_name not in inputs_moved_out, "currently cannot use both cur + prev frame"
      else:
        layer_name = name
        prev = False
        assert "prev:%s" % layer_name not in inputs_moved_out, "currently cannot use both cur + prev frame"
      assert layer_name in self.input_layers_moved_out
      assert isinstance(self.input_layers_net, TFNetwork)
      layer = self.input_layers_net.layers[layer_name]
      assert isinstance(layer, LayerBase)
      output = layer.output.copy_template_excluding_time_dim()
      with tf.name_scope("%s_moved_input" % name.replace(":", "_")):
        if prev:
          output.placeholder = tf.cond(
            tf.equal(i, 0),
            lambda: self._get_init_output(layer_name),
            lambda: inputs_moved_out_tas[layer_name].read(i - 1))
        else:
          output.placeholder = inputs_moved_out_tas[layer_name].read(i)
      layer = self.net.add_layer(name=name, output=output, layer_class=InternalLayer)
      inputs_moved_out[name] = layer
      return layer

    def get_layer(name):
      """
      :param str name: layer name
      :rtype: LayerBase
      """
      if name.startswith("prev:"):
        sub_name = name[len("prev:"):]
        if sub_name in self.input_layers_moved_out:
          return get_input_moved_out(name)
        return prev_layers[sub_name]
      if name.startswith("base:"):
        if name in extended_layers:
          return extended_layers[name]
        layer = self.parent_net.get_layer(name[len("base:"):])
        if self.parent_net.search_flag:
          if needed_beam_size:
            assert not layer.output.beam_size
            if layer.output.beam_size != needed_beam_size:
              layer = self.net.add_layer(
                name="%s_beam_%i" % (name, needed_beam_size),
                output=layer.output.copy_extend_with_beam(needed_beam_size),
                layer_class=InternalLayer)
              extended_layers[name] = layer
          assert layer.output.beam_size == needed_beam_size
        return layer
      if name in self.input_layers_moved_out:
        return get_input_moved_out(name)
      if name in self.output_layers_moved_out:
        # Will be constructed later.
        # This should not be used recursively, because we checked that nothing depends on it,
        # thus it should not be a problem to return None.
        return None
      return self.net.construct_layer(net_dict, name=name, get_layer=get_layer)

    # Go through needed_outputs, e.g. "output".
    # And prev_layers_needed because they might not be resolved otherwise.
    for layer_name in sorted(needed_outputs) + sorted(self.prev_layers_needed):
      if layer_name in self.input_layers_moved_out + self.output_layers_moved_out:
        continue
      get_layer(layer_name)
      assert layer_name in self.net.layers

  def _get_init_output(self, name):
    """
    :param str name: layer name
    :rtype: tf.Tensor
    """
    template_layer = self.layer_data_templates[name]
    cl = template_layer.layer_class_type
    assert issubclass(cl, LayerBase)
    batch_dim = template_layer.get_batch_dim()
    if name == "end" and template_layer.kwargs.get("initial_output", None) is None:
      # Special case for the 'end' layer.
      from TFUtil import constant_with_shape
      return constant_with_shape(False, shape=[batch_dim], name="initial_end")
    with reuse_name_scope(self.parent_rec_layer._rec_scope):
      with cl.cls_layer_scope(name):
        return cl.get_rec_initial_output(
          batch_dim=batch_dim, rec_layer=self.parent_rec_layer, **self.layer_data_templates[name].kwargs)

  def _get_init_extra_outputs(self, name):
    """
    :param str name: layer name
    :rtype: tf.Tensor|tuple[tf.Tensor]
    """
    template_layer = self.layer_data_templates[name]
    cl = template_layer.layer_class_type
    assert issubclass(cl, LayerBase)
    with reuse_name_scope(self.parent_rec_layer._rec_scope):
      with cl.cls_layer_scope(name):
        batch_dim = template_layer.get_batch_dim()
        d = cl.get_rec_initial_extra_outputs(
          batch_dim=batch_dim, rec_layer=self.parent_rec_layer, **self.layer_data_templates[name].kwargs)
    return d

  def _check_output_template_shape(self):
    output_template = self.layer_data_templates["output"]
    assert output_template.output.dim == self.parent_rec_layer.output.dim
    assert self.parent_rec_layer.output.time_dim_axis == 0
    assert output_template.output.time_dim_axis is None
    assert output_template.output.batch_shape == self.parent_rec_layer.output.batch_shape[1:], (
      "see RecLayer.get_out_data_from_opts()")

  def get_init_loop_vars(self):
    """
    :return: initial loop_vars. see self.get_next_loop_vars(). used in the body inside self.get_output()
    :rtype: (list[tf.Tensor],list[list[tf.Tensor]])
    """
    self._initial_outputs = {
      k: self._get_init_output(k)
      for k in sorted(self.prev_layers_needed)
      if k not in self.input_layers_moved_out + self.output_layers_moved_out}
    self._initial_extra_outputs = {
      k: self._get_init_extra_outputs(k)
      for k in sorted(self.layer_data_templates.keys())
      if k not in self.input_layers_moved_out + self.output_layers_moved_out}
    self._initial_extra_outputs = {k: v for (k, v) in self._initial_extra_outputs.items() if v}
    from Util import sorted_values_from_dict
    init_outputs_flat = sorted_values_from_dict(self._initial_outputs)
    init_extra_flat = [sorted_values_from_dict(v) for (k, v) in sorted(self._initial_extra_outputs.items())]
    return init_outputs_flat, init_extra_flat

  def get_init_loop_vars_shape_invariants(self):
    """
    :return: shape invariants, nested structure like get_init_loop_vars
    :rtype: (list[tf.TensorShape],list[tf.TensorShape|tuple[tf.TensorShape]])
    """
    assert self._initial_outputs is not None
    assert self._initial_extra_outputs is not None
    init_out_shapes = {
      k: tf.TensorShape(self.layer_data_templates[k].output.batch_shape)
      for k in self._initial_outputs}
    from TFUtil import nested_get_shapes
    init_rec_extra_shapes = nested_get_shapes(self._initial_extra_outputs)
    for name, shapes in init_rec_extra_shapes.items():
      # See also _get_init_extra_outputs.
      template_layer = self.layer_data_templates[name]
      cl = template_layer.layer_class_type
      d = cl.get_rec_initial_extra_outputs_shape_invariants(**self.layer_data_templates[name].kwargs)
      for k, shape in d.items():
        assert k in shapes
        # Not merge but replace because we intentionally want to allow relaxation.
        shapes[k] = shape
    from Util import sorted_values_from_dict
    init_outputs_flat = sorted_values_from_dict(init_out_shapes)
    init_extra_flat = [sorted_values_from_dict(v) for (k, v) in sorted(init_rec_extra_shapes.items())]
    return init_outputs_flat, init_extra_flat

  def get_layer_rec_var_from_loop_vars(self, loop_vars, layer_name):
    """
    :param (list[tf.Tensor],list[tf.Tensor]) loop_vars: loop_vars like in self.get_next_loop_vars()
    :param str layer_name:
    :return: layer rec_vars_outputs
    :rtype: dict[str,tf.Tensor]
    """
    prev_outputs_flat, prev_extra_flat = loop_vars
    assert len(prev_outputs_flat) == len(self._initial_outputs)
    assert len(prev_extra_flat) == len(self._initial_extra_outputs)
    from Util import dict_zip
    prev_extra = {
      k: dict_zip(sorted(self._initial_extra_outputs[k]), v)
      for (k, v) in zip(sorted(self._initial_extra_outputs), prev_extra_flat)}
    return prev_extra[layer_name]

  def get_parent_deps(self):
    """
    :return: list of dependencies to the parent network
    :rtype: list[LayerBase]
    """
    l = []
    layers = self.net.layers
    if not layers:  # happens only during initialization
      layers = self.layer_data_templates
    for _, layer in sorted(layers.items()):
      assert isinstance(layer, LayerBase)
      for dep in layer.get_dep_layers():
        # Usually dep.network is self.cell.net but it could reference to our own net,
        # e.g. if this is an attention layer like
        # {"class": "dot_attention", "base": "base:encoder", ...}.
        if dep.network is self.parent_net:
          if dep not in l:
            l += [dep]
    return l

  def get_output(self, rec_layer):
    """
    :param RecLayer rec_layer:
    :return: output of shape (time, batch, dim), (loss, error, loss_norm_factor), search choices
    :rtype: (tf.Tensor, (tf.Tensor, tf.Tensor, tf.Tensor), SearchChoices)
    """
    self._check_output_template_shape()
    from TFUtil import check_input_dim

    with tf.name_scope("subnet_base"):
      batch_dim = rec_layer.network.get_data_batch_dim()
      input_beam_size = None  # type: int | None
      if rec_layer.input_data:
        with tf.name_scope("x_tensor_array"):
          x, input_seq_len = rec_layer._get_input()  # x will be (time,batch,..,dim)
          x_shape = tf.shape(x, name="x_shape")
          x_ta = tf.TensorArray(
            name="x_ta",
            dtype=rec_layer.input_data.dtype,
            element_shape=tf.TensorShape(rec_layer.input_data.copy_template_excluding_time_dim().batch_shape),
            size=x_shape[0],
            infer_shape=True)
          x_ta = x_ta.unstack(x, name="x_ta_unstack")
        input_search_choices = rec_layer.network.get_search_choices(sources=rec_layer.sources)
        if input_search_choices:
          input_beam_size = input_search_choices.search_choices.beam_size
          assert self.parent_rec_layer.input_data.beam_size == input_beam_size
      else:
        x_ta = None
        input_seq_len = None
      if rec_layer.output.size_placeholder and not self.parent_net.search_flag:
        # See LayerBase._post_init_output(). could be set via target or size_target...
        # This should only be the case in training.
        fixed_seq_len = rec_layer.output.size_placeholder[0]
      else:
        fixed_seq_len = None
      if fixed_seq_len is None and "end" not in self.layer_data_templates:
        # If 'end' layer is not existing, the length must be defined.
        # In some cases (training with given target) we know the target sequence length.
        # Otherwise, by convention, it is defined by the input length
        # (assuming that there is an input which we iterate over).
        assert input_seq_len is not None, "length is not defined. provide an 'end' layer"
        fixed_seq_len = input_seq_len
      if fixed_seq_len is not None:
        with tf.name_scope("check_seq_len_batch_size"):
          fixed_seq_len = check_input_dim(fixed_seq_len, axis=0, dim=batch_dim * (input_beam_size or 1))
        max_seq_len = tf.reduce_max(fixed_seq_len, name="max_seq_len")
        have_known_seq_len = True
      else:
        assert "end" in self.layer_data_templates, "length not defined, provide 'end' layer"
        max_seq_len = None
        have_known_seq_len = False
      # if not self.input_data and self.network.search_flag:
      #   assert not have_known_seq_len  # at least for the moment

      # TODO: Better check for train_flag.
      # Maybe more generic via sampling options later.
      y_ta = None
      if rec_layer.target and ((
            rec_layer.network.train_flag is not False and not self.parent_net.search_flag) or rec_layer._cheating):
        # TODO check subnet, which extern data keys are used...
        y_data = rec_layer.network.get_extern_data(rec_layer.target, mark_data_key_as_used=True)
        y = y_data.get_placeholder_as_time_major()
        with tf.name_scope("y_max_len"):
          y_max_len = tf.shape(y)[0]
          if input_seq_len is not None:
            with tf.control_dependencies([tf.assert_equal(tf.reduce_max(input_seq_len), y_max_len,
                ["RecLayer %r with sources %r." % (rec_layer.name, rec_layer.sources),
                 " The length of the sources (", tf.reduce_max(input_seq_len),
                 ") differ from the length of the target (", y_max_len, ")."])]):
              y_max_len = tf.identity(y_max_len)
        y_ta = tf.TensorArray(
          name="y_ta",
          dtype=y_data.dtype,
          element_shape=tf.TensorShape(y_data.copy_template_excluding_time_dim().batch_shape),
          size=y_max_len,
          infer_shape=True)
        y_ta = y_ta.unstack(y, name="y_ta_unstack")
        if max_seq_len is None:
          max_seq_len = y_max_len

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

      if have_known_seq_len:
        min_loop_len = max_seq_len
      else:
        min_loop_len = 0

      from collections import namedtuple
      OutputToAccumulate = namedtuple("OutputToAccumulate", ["name", "dtype", "element_shape", "get"])
      outputs_to_accumulate = []  # type: list[OutputToAccumulate]
      needed_outputs = {"output"}

      def add_output_to_acc(layer_name):
        name = "output_%s" % layer_name
        if any([(out.name == name) for out in outputs_to_accumulate]):
          return
        outputs_to_accumulate.append(OutputToAccumulate(
          name=name,
          dtype=self.layer_data_templates[layer_name].output.dtype,
          element_shape=self.layer_data_templates[layer_name].output.batch_shape,
          get=lambda: self.net.layers[layer_name].output.placeholder))

      layer_names_with_losses = []
      if rec_layer.network.eval_flag:  # only collect losses if we need them
        # Note about the subnet loss calculation:
        # 1. We can collect the output and calculate the loss on the whole sequence.
        # 2. We can calculate the loss on a frame base and collect it per frame.
        # We implemented option 1 (collect output, loss on sequence) earlier.
        # Option 1 had the following disadvantages:
        # - It can require a lot of extra memory if the output is large,
        #   e.g. with a softmax output of 30k classes.
        # - The loss calculation can be numerical unstable, e.g. for cross-entropy.
        #   This could be solved by also storing the output before the activation (e.g. softmax),
        #   which would require even more memory, and other cases is wasted (e.g. MSE loss).
        #   There is no good way to determine in advance if we need it or not.
        # Option 2 has the disadvantage that some part of the code might be more hacky.
        # Overall, option 2 is more straight-forward, probably more what the user intends,
        # can use numerical stable variants (e.g. for cross-entropy + softmax),
        # and is what we do now.

        # Not so nice but simple way to get all relevant layers:
        layer_names_with_losses = [
          layer.name for layer in self.layer_data_templates.values()
          if layer.kwargs.get("loss") and not layer.kwargs.get("loss_only_on_non_search")]
        needed_outputs.update(layer_names_with_losses)

      # For search:
      # We will collect the search choices of the beam search,
      # to be able to reconstruct the final hypotheses.
      output_beam_size = None
      collected_choices = []  # type: list[str]  # layer names
      if rec_layer.network.search_flag:
        for layer in self.layer_data_templates.values():
          assert isinstance(layer, _TemplateLayer)
          if layer.search_choices:
            collected_choices += [layer.name]
            def get_derived(name):
              def get_choice_source_batches():
                layer = self.net.layers[name]
                return layer.search_choices.src_beams
              return get_choice_source_batches
            outputs_to_accumulate += [
              OutputToAccumulate(
                name="choice_%s" % layer.name,
                dtype=tf.int32,
                element_shape=(None, layer.search_choices.beam_size),  # (batch, beam)
                get=get_derived(layer.name))]

        if collected_choices:
          output_beam_size = self.layer_data_templates["output"].get_search_beam_size()
          assert output_beam_size is not None
          if fixed_seq_len is not None:
            assert input_beam_size in (1, None)
            from TFUtil import tile_transposed
            fixed_seq_len = tile_transposed(fixed_seq_len, axis=0, multiples=output_beam_size)  # (batch * beam,)

      if not have_known_seq_len:
        assert "end" in self.layer_data_templates, "You need to have an 'end' layer in your rec subnet."
        end_template = self.layer_data_templates["end"]
        needed_outputs.add("end")
        assert tf.as_dtype(end_template.output.dtype) is tf.bool
        assert end_template.output.batch_shape == (None,)  # (batch*beam,)
      else:
        assert have_known_seq_len, (
          "You need to have an 'end' layer in your rec subnet if the generated seq len is unknown.")

      if self.parent_rec_layer._optimize_move_layers_out:
        self._move_outside_loop(needed_outputs=needed_outputs)
      else:
        self.layers_in_loop = sorted(self.layer_data_templates.keys())

      if layer_names_with_losses:
        def make_get_loss_in_loop_frame(layer_name, return_error=False, return_loss=False):
          """
          :param str layer_name:
          :param bool return_error:
          :param bool return_loss:
          :rtype: ()->tf.Tensor
          """
          from TFNetworkLayer import Loss
          from TFUtil import identity

          def get_loss():
            layer = self.net.layers[layer_name]
            assert isinstance(layer.loss, Loss)
            # This is a bit hacky but we want to not reduce the loss to a scalar
            # in the loop but get it as shape (batch,).
            # This should work with all current implementations
            # but might need some redesign later.
            layer.loss.reduce_func = identity
            if return_loss:
              value = layer.get_loss_value()
            elif return_error:
              value = layer.get_error_value()
            else:
              assert False, "return_error or return_loss"
            assert isinstance(value, tf.Tensor)
            value.set_shape(tf.TensorShape((None,)))  # (batch,)
            return value

          return get_loss

        for layer_name in layer_names_with_losses:
          if layer_name not in self.layers_in_loop:
            continue  # will get loss out of them below
          outputs_to_accumulate.append(OutputToAccumulate(
            name="loss_%s" % layer_name,
            dtype=tf.float32,
            element_shape=(None,),  # (batch,)
            get=make_get_loss_in_loop_frame(layer_name, return_loss=True)))
          outputs_to_accumulate.append(OutputToAccumulate(
            name="error_%s" % layer_name,
            dtype=tf.float32,
            element_shape=(None,),  # (batch,)
            get=make_get_loss_in_loop_frame(layer_name, return_error=True)))

      if "output" in self.layers_in_loop:
        add_output_to_acc("output")

      # if a layer declares it is a output, we should save the values as well
      for name, template in self.layer_data_templates.items():
        if template.is_output_layer() and name not in needed_outputs:
          needed_outputs.add(name)
          add_output_to_acc(name)


      # Maybe some of the moved-out output-layers depend on data inside the loop,
      # so we should accumulate it to have access to it.
      for layer_name in self.output_layers_moved_out:
        for dep in self.layer_data_templates[layer_name].dependencies:
          if dep.name not in self.layers_in_loop:
            continue
          # Dependency is inside the loop, and we are using it, so we need to accumulate its output.
          add_output_to_acc(dep.name)
          needed_outputs.add(dep.name)

      # Tensor arrays for any layers which were moved out.
      input_layers_moved_out_tas = {}
      if self.input_layers_moved_out:
        with tf.name_scope("input_layers_moved_out"):
          self._construct_input_layers_moved_out()
          for layer_name in self.input_layers_moved_out:
            # Create only Tensor arrays for those which we use inside the loop.
            if not self._input_layer_used_inside_loop(layer_name):
              continue
            assert fixed_seq_len is not None
            inp_ta = tf.TensorArray(
              name="%s_ta" % layer_name,
              dtype=self.layer_data_templates[layer_name].output.dtype,
              element_shape=self.layer_data_templates[layer_name].output.batch_shape,
              size=tf.reduce_max(fixed_seq_len),
              infer_shape=True)
            inp_ta = inp_ta.unstack(
              self.input_layers_net.layers[layer_name].output.get_placeholder_as_time_major(),
              name="%s_ta_unstack" % layer_name)
            input_layers_moved_out_tas[layer_name] = inp_ta

      # Create a tensor array to store the intermediate values for each step i, e.g. of shape (batch, dim).
      init_acc_tas = [
        tf.TensorArray(
          name="acc_ta_%s" % out.name,
          dtype=out.dtype,
          element_shape=tf.TensorShape(out.element_shape),
          size=min_loop_len,
          dynamic_size=True,  # we will automatically grow it when needed
          infer_shape=True)
        for out in outputs_to_accumulate]

    def body(i, net_vars, acc_tas, seq_len_info=None):
      """
      The loop body of scan.

      :param tf.Tensor i: loop counter, scalar
      :param net_vars: the accumulator values. see also self.get_init_loop_vars()
      :param list[tf.TensorArray] acc_tas: the output accumulator TensorArray
      :param (tf.Tensor,tf.Tensor)|None seq_len_info: tuple (end_flag, seq_len)
      :return: [i + 1, a_flat, tas]: the updated counter + new accumulator values + updated TensorArrays
      :rtype: (tf.Tensor, object, list[tf.TensorArray])

      Raises:
        TypeError: if initializer and fn() output structure do not match
        ValueType: if initializer and fn() output lengths do not match
      """
      # The inner scope name is a bit screwed up and this is nicer anyway.
      with reuse_name_scope(rec_layer._rec_scope.name + "/while_loop_body", absolute=True):
        self.net.set_rec_step_info(
          i, end_flag=seq_len_info[0] if seq_len_info else None, seq_lens=fixed_seq_len)
        # get next loop vars (net_vars)
        from TFUtil import identity_op_nested, select_src_beams
        from Util import sorted_values_from_dict, dict_zip
        prev_outputs_flat, prev_extra_flat = net_vars
        assert len(prev_outputs_flat) == len(self._initial_outputs)  # subset of self.prev_layers_needed
        prev_outputs = {k: v for (k, v) in zip(sorted(self._initial_outputs), prev_outputs_flat)}
        with tf.name_scope("prev_outputs"):
          prev_outputs = identity_op_nested(prev_outputs)
        assert len(prev_extra_flat) == len(self._initial_extra_outputs)
        prev_extra = {
          k: dict_zip(sorted(self._initial_extra_outputs[k]), v)
          for (k, v) in zip(sorted(self._initial_extra_outputs), prev_extra_flat)}
        with tf.name_scope("prev_extra"):
          prev_extra = identity_op_nested(prev_extra)
        with reuse_name_scope(self.parent_rec_layer._rec_scope):
          self._construct(
            prev_outputs=prev_outputs, prev_extra=prev_extra,
            i=i,
            data=x_ta.read(i, name="x_ta_read") if x_ta else None,
            classes=y_ta.read(i, name="y_ta_read") if y_ta else None,
            inputs_moved_out_tas=input_layers_moved_out_tas,
            needed_outputs=needed_outputs)

        transformed_cache = {}  # layer -> layer

        def maybe_transform(layer):
          """
          This will be available in the next loop frame as the "prev:..." layer.
          If the current search choices are already from the prev frame, select beams such that we end up
          in the current frame.
          :param LayerBase layer:
          :rtype: LayerBase
          """
          if not self.parent_net.search_flag:
            return layer
          if layer in transformed_cache:
            return transformed_cache[layer]
          assert not RecLayer.is_prev_step_layer(layer)  # this layer is from current frame
          search_choices_layer = layer.get_search_choices().owner
          if not RecLayer.is_prev_step_layer(search_choices_layer):
            return layer
          assert search_choices_layer.name.startswith("prev:")
          cur_frame_search_choices_layer = self.net.layers[search_choices_layer.name[len("prev:"):]]
          assert not RecLayer.is_prev_step_layer(cur_frame_search_choices_layer)
          transformed_layer = cur_frame_search_choices_layer.search_choices.translate_to_this_search_beam(layer)
          assert transformed_layer != layer
          transformed_cache[layer] = transformed_layer
          return transformed_layer
        outputs_flat = [
          maybe_transform(self.net.layers[k]).output.placeholder for k in sorted(self._initial_outputs)]
        extra_flat = [
          sorted_values_from_dict(maybe_transform(self.net.layers[k]).rec_vars_outputs)
          for k in sorted(self._initial_extra_outputs)]
        net_vars = (outputs_flat, extra_flat)

        if seq_len_info is not None:
          end_flag, dyn_seq_len = seq_len_info
          choices = self.net.layers["end"].get_search_choices()
          assert choices, "no search choices in layer %r" % self.net.layers["end"]
          with tf.name_scope("end_flag"):
            end_flag = select_src_beams(end_flag, src_beams=choices.src_beams)
            end_flag = tf.logical_or(end_flag, self.net.layers["end"].output.placeholder)  # (batch * beam,)
          with tf.name_scope("dyn_seq_len"):
            dyn_seq_len = select_src_beams(dyn_seq_len, src_beams=choices.src_beams)
            dyn_seq_len += tf.where(
              end_flag,
              constant_with_shape(0, shape=tf.shape(end_flag)),
              constant_with_shape(1, shape=tf.shape(end_flag)))  # (batch * beam,)
            seq_len_info = (end_flag, dyn_seq_len)
        assert len(acc_tas) == len(outputs_to_accumulate)
        acc_tas = [
          acc_ta.write(i, out.get(), name="%s_acc_ta_write" % out.name)
          for (acc_ta, out) in zip(acc_tas, outputs_to_accumulate)]
        next_i = tf.add(i, 1, name="next_i")
        res = (next_i, net_vars, acc_tas)
        if seq_len_info is not None:
          res += (seq_len_info,)
        if self._debug_out is not None:
          from TFUtil import identity_with_debug_log
          args = {"step": i}
          args.update({"%s.output" % k: v.output.placeholder for (k, v) in self.net.layers.items()})
          for k in self._initial_extra_outputs:
            args.update({"%s.extra.%s" % (k, k2): v for (k2, v) in self.net.layers[k].rec_vars_outputs.items()})
            args.update({"prev:%s.extra.%s" % (k, k2): v for (k2, v) in prev_extra[k].items()})
          res = (identity_with_debug_log(out=self._debug_out, x=res[0], args=args),) + res[1:]
        return res

    def cond(i, net_vars, acc_tas, seq_len_info=None):
      """
      :param tf.Tensor i: loop counter, scalar
      :param net_vars: the accumulator values. see also self.get_init_loop_vars()
      :param list[tf.TensorArray] acc_tas: the output accumulator TensorArray
      :param (tf.Tensor,tf.Tensor)|None seq_len_info: tuple (end_flag, seq_len)
      :return: True -> we should run the current loop-iteration, False -> stop loop
      :rtype: tf.Tensor
      """
      with tf.name_scope("loop_cond"):
        from TFUtil import opt_logical_and
        res = True
        if max_seq_len is not None:
          res = opt_logical_and(res, tf.less(i, max_seq_len, name="i_less_max_seq_len"))
        # Only consider the user 'max_seq_len' option if we don't know the real max_seq_len.
        # This is the old behavior. Maybe this might change at some point.
        elif isinstance(rec_layer._max_seq_len, (int, tf.Tensor)):
          res = opt_logical_and(res, tf.less(i, rec_layer._max_seq_len, name="i_less_max_seq_len"))
        else:
          assert rec_layer._max_seq_len is None, "%r: unsupported max_seq_len %r" % (rec_layer, rec_layer._max_seq_len)
        # Check not considering seq_len_info because the dynamics of the network can also lead
        # to an infinite loop, so enforce that some maximum is specified.
        assert res is not True, "%r: specify max_seq_len" % rec_layer
        if seq_len_info is not None:
          end_flag, _ = seq_len_info
          any_not_ended = tf.reduce_any(tf.logical_not(end_flag), name="any_not_ended")
          res = opt_logical_and(res, any_not_ended)
        return res

    from TFUtil import constant_with_shape
    init_loop_vars = (
      tf.constant(0, name="initial_i"),
      self.get_init_loop_vars(),
      init_acc_tas)
    shape_invariants = (
      tf.TensorShape(()),
      self.get_init_loop_vars_shape_invariants(),
      [tf.TensorShape(None) for _ in init_acc_tas])
    if not have_known_seq_len:
      # See body().
      out_batch_dim = self.layer_data_templates["end"].get_batch_dim()
      init_seq_len_info = (
        constant_with_shape(False, shape=[out_batch_dim], name="initial_end_flag"),
        constant_with_shape(0, shape=[out_batch_dim], name="initial_seq_len"))
      init_loop_vars += (init_seq_len_info,)
      shape_invariants += ((tf.TensorShape([None]), tf.TensorShape([None])),)
    if self.layers_in_loop:
      final_loop_vars = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=init_loop_vars,
        shape_invariants=shape_invariants,
        back_prop=self.net.train_flag is not False)
    else:  # no layers inside loop, all optimized out
      final_loop_vars = init_loop_vars
    if have_known_seq_len:
      assert fixed_seq_len is not None
      seq_len = fixed_seq_len
      _, final_net_vars, final_acc_tas = final_loop_vars
    else:
      _, final_net_vars, final_acc_tas, (_, seq_len) = final_loop_vars
    if rec_layer.output.size_placeholder is None:
      rec_layer.output.size_placeholder = {}
    rec_layer.output.size_placeholder[0] = seq_len
    assert isinstance(final_acc_tas, list)
    if len(outputs_to_accumulate) > 0:
      assert isinstance(final_acc_tas[0], tf.TensorArray)
    assert len(final_acc_tas) == len(outputs_to_accumulate)
    self.final_acc_tas_dict = {
      out.name: final_acc_ta
      for (final_acc_ta, out) in zip(final_acc_tas, outputs_to_accumulate)}  # type: dict[str,tf.TensorArray]

    self._construct_output_layers_moved_out(loop_accumulated=self.final_acc_tas_dict, seq_len=seq_len)

    sub_loss = sub_error = sub_loss_normalization_factor = None
    if layer_names_with_losses:
      with tf.name_scope("sub_net_loss"):
        with tf.name_scope("sub_loss_normalization_factor"):
          sub_loss_normalization_factor = 1.0 / tf.cast(tf.reduce_sum(seq_len), tf.float32)
        sub_loss = 0.0  # accumulated
        for layer_name in sorted(layer_names_with_losses):
          if layer_name in self.input_layers_moved_out + self.output_layers_moved_out:
            if layer_name in self.input_layers_moved_out:
              layer_with_loss_inst = self.input_layers_net.layers[layer_name]
            else:
              layer_with_loss_inst = self.output_layers_net.layers[layer_name]
            assert isinstance(layer_with_loss_inst, LayerBase)
            with reuse_name_scope(layer_with_loss_inst.tf_scope_name):
              loss_value = layer_with_loss_inst.get_loss_value()
              error_value = layer_with_loss_inst.get_error_value()
            sub_loss += loss_value * layer_with_loss_inst.loss_scale
            # Only one error, not summed up. Determined by sorted layers.
            sub_error = error_value
          else:
            layer_with_loss_inst = self.net.layers[layer_name]
            loss_value = self.final_acc_tas_dict["loss_%s" % layer_name].stack(name="loss_%s_stack" % layer_name)
            error_value = self.final_acc_tas_dict["error_%s" % layer_name].stack(name="error_%s_stack" % layer_name)
            loss_value.set_shape(tf.TensorShape((None, None)))  # (time, batch)
            error_value.set_shape(tf.TensorShape((None, None)))  # (time, batch)

            from TFUtil import sequence_mask_time_major
            mask = sequence_mask_time_major(seq_len)
            loss_value = tf.where(mask, loss_value, tf.zeros_like(loss_value))
            error_value = tf.where(mask, error_value, tf.zeros_like(error_value))
            loss_value = tf.reduce_sum(loss_value)
            error_value = tf.reduce_sum(error_value)

            sub_loss += loss_value * layer_with_loss_inst.loss_scale
            # Only one error, not summed up. Determined by sorted layers.
            sub_error = error_value

    # Check if collected_choices has all the right layers.
    # At the moment, _TemplateLayer.has_search_choices() might be incomplete, that is why we check here.
    for layer in self.net.layers.values():
      if layer.name.startswith("prev:"):
        continue
      if layer.search_choices:
        assert layer.name in collected_choices
    for name in collected_choices:
      layer = self.net.layers[name]
      assert layer.search_choices

    search_choices = None
    if collected_choices:
      # Find next choice layer. Then iterate through its source choice layers through time
      # and resolve the output over time to be in line with the final output search choices.
      output_choice_base = self.net.get_search_choices(src=rec_layer.cell.net.layers["output"])
      assert isinstance(output_choice_base, LayerBase)
      assert output_beam_size == output_choice_base.search_choices.beam_size
      initial_beam_choices = tf.range(0, output_beam_size)  # (beam_out,)
      from TFUtil import expand_dims_unbroadcast
      initial_beam_choices = expand_dims_unbroadcast(
        initial_beam_choices, axis=0, dim=batch_dim)  # (batch, beam_out)

      new_acc_output_ta = tf.TensorArray(
        name="new_acc_output_ta",
        dtype=self.layer_data_templates["output"].output.dtype,
        element_shape=tf.TensorShape(self.layer_data_templates["output"].output.batch_shape),
        size=final_acc_tas[0].size(),
        infer_shape=True)

      # TODO: Maybe we could use tf.contrib.seq2seq.GatherTree op here instead...
      def search_resolve_body(i, choice_beams, new_acc_output_ta):
        # This loops goes backwards through time.
        # This starts at i == seq_len - 1.
        # choice_beams are from the previous step, shape (batch, beam_out) -> beam idx of output,
        # output is of shape (batch * beam, n_out).
        with reuse_name_scope(rec_layer._rec_scope.name + "/while_loop_search_body", absolute=True):
          # We start at the output layer choice base, and search for its source, i.e. for the previous time frame.
          choice_base = output_choice_base
          is_output_choice = True
          while True:
            assert choice_base.network is rec_layer.cell.net, "not yet implemented otherwise"

            src_choice_beams = self.final_acc_tas_dict["choice_%s" % choice_base.name].read(i)  # (batch, beam) -> beam_in idx
            assert src_choice_beams.get_shape().ndims == 2

            with tf.name_scope("choice_beams"):
              from TFUtil import nd_indices, assert_min_tf_version
              assert_min_tf_version((1, 1), "gather_nd")
              idxs_exp = nd_indices(choice_beams)  # (batch, beam_out, 2) -> (batch idx, beam idx)
              src_choice_beams = tf.gather_nd(src_choice_beams, idxs_exp)  # (batch, beam_out)
            if is_output_choice:
              with tf.name_scope("output"):
                output = self.final_acc_tas_dict["output_output"].read(i)  # (batch * beam, [n_out])
                out_shape = list(rec_layer.output.batch_shape[1:])  # without time-dim
                output.set_shape(tf.TensorShape(out_shape))
                output = tf.reshape(
                  output,
                  [batch_dim,
                   output_beam_size] + out_shape[1:])  # (batch, beam, [n_out])
                output = tf.gather_nd(output, idxs_exp)  # (batch, beam_par, [n_out])
                output = tf.reshape(
                  output,
                  [batch_dim * output_beam_size] + out_shape[1:])  # (batch * beam_par, [n_out])
                new_acc_output_ta = new_acc_output_ta.write(i, output)

            assert choice_base.search_choices
            src_choice_layer = choice_base.search_choices.src_layer
            assert src_choice_layer is not None  # must be one, e.g. from prev time frame
            if isinstance(src_choice_layer, _TemplateLayer):
              assert src_choice_layer.is_prev_time_frame
              return (
                i - 1,
                src_choice_beams,
                new_acc_output_ta)
            is_output_choice = False
            choice_base = src_choice_layer
            choice_beams = src_choice_beams

      _, _, new_acc_output_ta = tf.while_loop(
        name="search_resolve_loop",
        cond=(lambda i, *args: tf.greater_equal(i, 0, name="search_resolve_loop_cond_i_ge_0")),
        body=search_resolve_body,
        loop_vars=(
          tf.identity(final_acc_tas[0].size() - 1, name="search_resolve_initial_i"),  # we go backwards
          initial_beam_choices,
          new_acc_output_ta),
        back_prop=self.net.train_flag is not False)
      self.final_acc_tas_dict["output_output"] = new_acc_output_ta

      # Collect the search choices for the rec layer itself.
      # Our output will be of shape (time, batch * beam, dim).
      # The beam scores will be of shape (batch, beam).
      search_choices = SearchChoices(owner=rec_layer, beam_size=output_beam_size)
      # TODO search_choices.src_beams, not really supported currently
      final_choice_rec_vars = self.get_layer_rec_var_from_loop_vars(
        loop_vars=final_net_vars,
        layer_name=output_choice_base.name)
      search_choices.set_beam_scores_from_rec(final_choice_rec_vars)

    with tf.name_scope("output"):
      if "output" in self.input_layers_moved_out:
        output = self.input_layers_net.layers["output"].output.get_placeholder_as_time_major()
      elif "output" in self.output_layers_moved_out:
        output = self.output_layers_net.layers["output"].output.get_placeholder_as_time_major()
      else:
        output = self.final_acc_tas_dict["output_output"].stack(name="output_stack")  # e.g. (time, batch, dim)
        if not have_known_seq_len:
          with tf.name_scope("output_sub_slice"):
            output = output[:tf.reduce_max(seq_len)]  # usually one less

    for key in self.net.used_data_keys | (self.input_layers_net.used_data_keys if self.input_layers_net else set()) | (self.output_layers_net.used_data_keys if self.output_layers_net else set()):
      if key == "source":
        continue
      self.parent_net.used_data_keys.add(key)

    return output, (sub_loss, sub_error, sub_loss_normalization_factor), search_choices

  def _input_layer_used_inside_loop(self, layer_name):
    """
    :param str layer_name:
    :return: whether the layer is used by any other layer inside the loop
    :rtype: bool
    """
    layer = self.layer_data_templates[layer_name]
    for layer_in_loop in self.layer_data_templates.values():
      if layer_in_loop.name in self.input_layers_moved_out:
        continue
      if layer_in_loop.name in self.output_layers_moved_out:
        continue
      if layer in layer_in_loop.dependencies:
        return True
    return False

  def _move_outside_loop(self, needed_outputs):
    """
    Based on the templated network, we can see the dependencies.
    We want to move as much calculation, i.e. subnet layers, as possible out of the loop.
    E.g. an (input) layer which does not depend on any output from the previous frame can be calculated in advance.
    And an (output) layer which is not used for other calculations inside the loop can be calculated out-of-the-loop.

    :param set[str] needed_outputs:
    :return: nothing, will set self.input_layers_moved_out/output_layers_moved_out/layers_in_loop
    """
    layers_in_loop = []  # type: list[_TemplateLayer]

    def visit(deps):
      """
      :param list[LayerBase] deps:
      """
      for l in deps:
        if not isinstance(l, _TemplateLayer):  # real layer from base net or so
          continue
        if l.name == "data" or l.name.startswith("data:"):
          continue
        assert self.layer_data_templates[l.name] is l
        if l not in layers_in_loop:
          layers_in_loop.append(l)
          visit(sorted(l.dependencies, key=lambda l: l.name))
    visit([self.layer_data_templates[name] for name in needed_outputs])

    self.input_layers_moved_out = []
    self.output_layers_moved_out = []

    def output_can_move_out(layer):
      assert isinstance(layer, _TemplateLayer)
      # Special case: end-layer, which is added if the seq-len is unknown, cannot be moved out.
      if layer.name == "end":
        return False
      if self.parent_net.search_flag:
        if issubclass(layer.layer_class_type, ChoiceLayer):
          return False  # need to perform the search inside the loop currently
      # layer.output is used by other layers?
      for other_layer in layers_in_loop:
        if layer in other_layer.get_dep_layers():
          return False
      return True

    def find_output_layer_to_move_out():
      for layer in layers_in_loop:
        if output_can_move_out(layer):
          return layer
      return None

    def output_move_out(layer):
      assert isinstance(layer, _TemplateLayer)
      layers_in_loop.remove(layer)
      self.output_layers_moved_out.append(layer.name)

    def input_can_move_out(layer):
      assert isinstance(layer, _TemplateLayer)
      if self.parent_net.search_flag:
        if issubclass(layer.layer_class_type, ChoiceLayer):
          return False  # need to perform the search inside the loop currently
      layer_deps = layer.get_dep_layers()
      # We depend on other layers from this sub-network?
      for other_layer in layers_in_loop:
        if other_layer in layer_deps:
          return False
      return True

    def find_input_layer_to_move_out():
      for layer in layers_in_loop:
        if input_can_move_out(layer):
          return layer
      return None

    def input_move_out(layer):
      assert isinstance(layer, _TemplateLayer)
      layers_in_loop.remove(layer)
      self.input_layers_moved_out.append(layer.name)

    while True:
      output_layer = find_output_layer_to_move_out()
      if output_layer:
        output_move_out(output_layer)
      input_layer = find_input_layer_to_move_out()
      if input_layer:
        input_move_out(input_layer)
      if not output_layer and not input_layer:
        break

    self.layers_in_loop = [layer.name for layer in layers_in_loop]

    log_stream = log.v3
    print("Rec layer sub net:", file=log_stream)
    remaining_layers = set(self.net_dict.keys())

    def dump_info(s, l):
      print("  %s: (#: %i)" % (s, len(l)), file=log_stream)
      for layer_name in l:
        print("    %s" % layer_name, file=log_stream)
        remaining_layers.remove(layer_name)
      if not l:
        print("    None", file=log_stream)

    dump_info("Input layers moved out of loop", self.input_layers_moved_out)
    dump_info("Output layers moved out of loop", self.output_layers_moved_out)
    dump_info("Layers in loop", self.layers_in_loop)
    dump_info("Unused layers", sorted(remaining_layers))

  def _construct_input_layers_moved_out(self):
    """
    See self._move_outside_loop().
    The input layers will be constructed in self.input_layers_net.

    :return: nothing, will init self.input_layers_net
    """
    if not self.input_layers_moved_out:
      return

    from TFNetwork import TFNetwork, ExternData
    from TFNetworkLayer import InternalLayer
    self.input_layers_net = TFNetwork(
      name="%s/%s:rec-subnet-input" % (self.parent_net.name, self.parent_rec_layer.name if self.parent_rec_layer else "?"),
      extern_data=ExternData(),
      train_flag=self.parent_net.train_flag,
      search_flag=self.parent_net.search_flag,
      parent_layer=self.parent_rec_layer,
      parent_net=self.parent_net)
    if self.parent_rec_layer.input_data:
      self.input_layers_net.extern_data.data["source"] = \
        self.parent_rec_layer.input_data
    for key in self.parent_net.extern_data.data.keys():
      self.input_layers_net.extern_data.data[key] = \
        self.parent_net.extern_data.data[key]

    def get_prev_layer(name):
      """
      :param str name: layer name without "prev:" prefix
      :rtype: LayerBase
      """
      cur_layer = get_layer(name)
      with tf.name_scope("prev_%s" % name):
        # See also _construct_output_layers_moved_out.
        output = cur_layer.output.copy_as_time_major()
        initial = self._get_init_output(name)
        initial_wt = tf.expand_dims(initial, axis=0)  # add time axis
        x = output.placeholder
        output.placeholder = tf.concat([initial_wt, x], axis=0, name="concat_in_time")
        output.placeholder = output.placeholder[:-1]  # remove last frame
        # Note: This seq_len might make sense to use here:
        # output.size_placeholder[0] = tf.minimum(output.size_placeholder[0] + 1, tf.shape(x)[0])
        # However, often we assume that we keep the same seq lens as the output layer.
        assert isinstance(self.input_layers_net, TFNetwork)
        layer = self.input_layers_net.add_layer(name="prev:%s" % name, output=output, layer_class=InternalLayer)
        return layer

    # get_layer similar as in self._construct() but simplified.
    def get_layer(name):
      """
      :param str name: layer name
      :rtype: LayerBase
      """
      assert isinstance(self.input_layers_net, TFNetwork)
      if name in self.input_layers_net.layers:
        return self.input_layers_net.layers[name]
      if name.startswith("prev:"):
        return get_prev_layer(name[len("prev:"):])
      if name.startswith("base:"):
        return self.parent_net.layers[name[len("base:"):]]
      return self.input_layers_net.construct_layer(self.net_dict, name=name, get_layer=get_layer)

    # Same scope as the main subnet, so that it stays compatible.
    with reuse_name_scope(self.parent_rec_layer._rec_scope):
      for layer_name in self.input_layers_moved_out:
        get_layer(layer_name)

  def _construct_output_layers_moved_out(self, loop_accumulated, seq_len):
    """
    See self._move_outside_loop().
    The output layers will be constructed in self.output_layers_net.

    :param dict[str,tf.TensorArray] loop_accumulated:
      keys, see self.get_output(). should be like "output_<layer_name>"
    :param tf.Tensor seq_len: shape (batch,)
    :return: nothing, will init self.output_layers_net
    """
    if not self.output_layers_moved_out:
      return

    from TFNetwork import TFNetwork, ExternData
    from TFNetworkLayer import InternalLayer
    self.output_layers_net = TFNetwork(
      name="%s/%s:rec-subnet-output" % (self.parent_net.name, self.parent_rec_layer.name if self.parent_rec_layer else "?"),
      extern_data=ExternData(),
      train_flag=self.parent_net.train_flag,
      search_flag=self.parent_net.search_flag,
      parent_layer=self.parent_rec_layer,
      parent_net=self.parent_net)
    if self.parent_rec_layer.input_data:
      self.output_layers_net.extern_data.data["source"] = \
        self.parent_rec_layer.input_data
    for key in self.parent_net.extern_data.data.keys():
      self.output_layers_net.extern_data.data[key] = \
        self.parent_net.extern_data.data[key]

    prev_layers = {}  # type: dict[str,InternalLayer]
    loop_acc_layers = {}  # type: dict[str,InternalLayer]

    def get_loop_acc_layer(name):
      """
      :param str name:
      :rtype: LayerBase
      """
      if name in loop_acc_layers:
        return loop_acc_layers[name]
      with tf.name_scope(self.layer_data_templates[name].layer_class_type.cls_get_tf_scope_name(name)):
        output = self.layer_data_templates[name].output.copy_template_adding_time_dim(time_dim_axis=0)
        # We should have accumulated it.
        output.placeholder = loop_accumulated["output_%s" % name].stack()  # e.g. (time,batch,dim)
        output.size_placeholder = {0: seq_len}
        assert isinstance(self.output_layers_net, TFNetwork)
        layer = self.output_layers_net.add_layer(name=name, output=output, layer_class=InternalLayer)
        loop_acc_layers[name] = layer
        return layer

    def get_prev_layer(name):
      """
      :param str name: excluding "prev:" prefix
      :rtype: LayerBase
      """
      if name in prev_layers:
        return prev_layers[name]
      cur_layer = get_layer(name)
      with tf.name_scope("prev_%s" % name):
        output = cur_layer.output.copy_as_time_major()
        initial = self._get_init_output(name)
        initial_wt = tf.expand_dims(initial, axis=0)  # add time axis
        x = output.placeholder
        output.placeholder = tf.concat([initial_wt, x], axis=0, name="concat_in_time")
        output.placeholder = output.placeholder[:-1]  # remove last frame
        # Note: This seq_len might make sense to use here:
        # output.size_placeholder[0] = tf.minimum(output.size_placeholder[0] + 1, tf.shape(x)[0])
        # However, often we assume that we keep the same seq lens as the output layer.
        output.size_placeholder[0] = seq_len
        assert isinstance(self.output_layers_net, TFNetwork)
        layer = self.output_layers_net.add_layer(name="prev:%s" % name, output=output, layer_class=InternalLayer)
        prev_layers[name] = layer
        return layer

    # get_layer similar as in self._construct() but simplified.
    def get_layer(name):
      """
      :param str name:
      :rtype: LayerBase
      """
      if name.startswith("prev:"):
        return get_prev_layer(name[len("prev:"):])
      if name.startswith("base:"):
        return self.parent_net.layers[name[len("base:"):]]
      if name in self.input_layers_moved_out:
        return self.input_layers_net.layers[name]
      if name not in self.output_layers_moved_out:
        # It means that the layer is inside the loop.
        return get_loop_acc_layer(name)
      return self.output_layers_net.construct_layer(self.net_dict, name=name, get_layer=get_layer)

    # Same scope as the main subnet, so that it stays compatible.
    with reuse_name_scope(self.parent_rec_layer._rec_scope):
      for layer_name in self.output_layers_moved_out:
        get_layer(layer_name)


class _TemplateLayer(LayerBase):
  """
  Used by _SubnetworkRecCell.
  In a first pass, it creates template layers with only the meta information about the Data.
  All "prev:" layers also stay instances of _TemplateLayer in the real computation graph.
  """

  def __init__(self, network, name):
    """
    :param TFNetwork.TFNetwork network:
    :param str name:
    """
    # Init with some dummy.
    super(_TemplateLayer, self).__init__(
      out_type={"shape": (), "name": "dummy_initial_template_data"},
      name=name, network=network)
    self.output.size_placeholder = {}  # must be initialized
    self.layer_class = ":uninitialized-template"
    self.is_data_template = False
    self.is_prev_time_frame = False
    self.layer_class_type = None  # type: type[LayerBase]|LayerBase
    self.kwargs = None  # type: dict[str]
    self.dependencies = set()  # type: set[LayerBase]
    self._template_base = None  # type: _TemplateLayer

  def __repr__(self):
    return "<%s(%s)(%s) %r out_type=%s>" % (
      self.__class__.__name__, self.layer_class_type.__name__ if self.layer_class_type else None, self.layer_class,
      self.name, self.output.get_description(with_name=False))

  def init(self, output, layer_class, template_type="template", **kwargs):
    """
    :param Data output:
    :param type[LayerBase]|LayerBase layer_class:
    :param str template_type:
    """
    # Overwrite self.__class__ so that checks like isinstance(layer, ChoiceLayer) work.
    # Not sure if this is the nicest way -- probably not, so I guess this will go away later.
    self.is_prev_time_frame = (template_type == "prev")
    self.is_data_template = (template_type == "template")
    assert self.is_prev_time_frame or self.is_data_template
    self.layer_class = ":%s:%s" % (template_type, layer_class.layer_class)
    self.output = output
    if not self.output.size_placeholder:
      self.output.size_placeholder = {}
    self.layer_class_type = layer_class
    self.kwargs = kwargs
    self.kwargs["output"] = output
    self._is_output_layer = kwargs.get("is_output_layer", None)
    if self._has_search_choices():
      self.search_choices = SearchChoices(owner=self, beam_size=self._get_search_choices_beam_size())

  def copy_as_prev_time_frame(self, prev_output=None, rec_vars_prev_outputs=None):
    """
    :param tf.Tensor|None prev_output:
    :param dict[str,tf.Tensor]|None rec_vars_prev_outputs:
    :return: new _TemplateLayer
    :rtype: _TemplateLayer
    """
    l = _TemplateLayer(network=self.network, name="prev:%s" % self.name)
    l._template_base = self
    l.dependencies = self.dependencies
    l.init(layer_class=self.layer_class_type, template_type="prev", **self.kwargs)
    if prev_output is not None:
      l.output.placeholder = prev_output
      l.output.placeholder.set_shape(tf.TensorShape(l.output.batch_shape))
      assert l.output.placeholder.dtype is tf.as_dtype(l.output.dtype)
      l.output.size_placeholder = {}  # must be set
    if rec_vars_prev_outputs is not None:
      l.rec_vars_outputs = rec_vars_prev_outputs
    if self.search_choices:
      l.search_choices = SearchChoices(owner=l, beam_size=self.search_choices.beam_size)
      l.search_choices.set_beam_scores_from_own_rec()
      l.output.beam_size = self.search_choices.beam_size
    return l

  def get_dep_layers(self):
    if self.is_data_template:
      # This is from the template construction, a layer in _SubnetworkRecCell.layer_data_templates.
      # Maybe we already have the layer constructed.
      real_layer = self.network.layers.get(self.name)
      if real_layer:
        return real_layer.get_dep_layers()
      # All refs to this subnet are other _TemplateLayer, no matter if prev-frame or not.
      # Otherwise, refs to the base network are given as-is.
      return sorted(self.dependencies, key=lambda l: l.name)
    assert self.is_prev_time_frame
    # In the current frame, the deps would be self.dependencies.
    # (It's ok that this would not contain prev-frames.)
    # We want to return the logical dependencies here, i.e. all such layers from previous frames.
    # Not all of them might exist, but then, we want to get their dependencies.
    cur_deps = sorted(self.dependencies, key=lambda l: l.name)
    deps = []
    for layer in cur_deps:
      if layer.network is not self.network:
        if layer not in deps:
          deps.append(layer)
        continue
      assert isinstance(layer, _TemplateLayer)
      assert layer.is_data_template
      # Find the related prev-frame layer.
      prev_layer = self.network.layers.get("prev:%s" % layer.name, None)
      if prev_layer:
        if prev_layer not in deps:
          deps.append(prev_layer)
        continue
      # Not yet constructed or not needed to construct.
      # In that case, add its dependencies instead.
      layer_deps = sorted(layer.dependencies, key=lambda l: l.name)
      for dep in layer_deps:
        if dep not in cur_deps:
          cur_deps.append(dep)  # the current iterable will also visit this
    return deps

  def _has_search_choices(self):
    """
    :return: whether an instance of this class has search_choices set
    :rtype: bool
    """
    # TODO: extend if this is a subnet or whatever
    if not self.network.search_flag:
      return False
    return issubclass(self.layer_class_type, ChoiceLayer)

  def _get_search_choices_beam_size(self):
    """
    Only valid if self.has_search_choices() is True.
    :rtype: int
    """
    return self.kwargs["beam_size"]

  def get_hidden_state(self):
    if issubclass(self.layer_class_type, RnnCellLayer):
      return self.rec_vars_outputs["state"]
    return super(_TemplateLayer, self).get_hidden_state()

  def get_last_hidden_state(self, key):
    if issubclass(self.layer_class_type, RnnCellLayer):
      return RnnCellLayer.get_state_by_key(self.rec_vars_outputs["state"], key=key)
    return super(_TemplateLayer, self).get_last_hidden_state(key=key)


class RecStepInfoLayer(LayerBase):
  """
  Used by _SubnetworkRecCell.
  Represents the current step number.
  """

  layer_class = ":i"

  def __init__(self, i, end_flag=None, seq_lens=None, **kwargs):
    """
    :param tf.Tensor i: scalar, int32, current step (time)
    :param tf.Tensor|None end_flag: (batch,), bool, says that the current sequence has ended
    :param tf.Tensor|None seq_lens: (batch,) int32, seq lens
    """
    super(RecStepInfoLayer, self).__init__(
      output=Data(name="i", shape=(), dtype="int32", sparse=False, placeholder=tf.expand_dims(i, axis=0)),
      **kwargs)
    self.step = i
    self.end_flag = end_flag
    self.seq_lens = seq_lens

  def get_end_flag(self):
    """
    :return: (batch,) of type bool. batch might include beam size
    :rtype: tf.Tensor
    """
    if self.end_flag is None:
      assert self.seq_lens is not None
      from TFUtil import reuse_name_scope_of_tensor
      with reuse_name_scope_of_tensor(self.step, postfix="/end_flag"):
        self.end_flag = tf.greater_equal(self.step, self.seq_lens)
    return self.end_flag


class RnnCellLayer(_ConcatInputLayer):
  """
  Wrapper around tf.contrib.rnn.RNNCell.
  This will operate a single step, i.e. there is no time dimension,
  i.e. we expect a (batch,n_in) input, and our output is (batch,n_out).
  This is expected to be used inside a RecLayer.
  """

  layer_class = "rnn_cell"

  def __init__(self, n_out, unit, unit_opts=None,
               initial_state=None, initial_output=None,
               weights_init="xavier", **kwargs):
    """
    :param int n_out: so far, only output shape (batch,n_out) supported
    :param str|tf.contrib.rnn.RNNCell unit: e.g. "BasicLSTM" or "LSTMBlock"
    :param dict[str]|None unit_opts: passed to the cell.__init__
    :param str|float|LayerBase|tuple[LayerBase]|dict[LayerBase] initial_state: see self.get_rec_initial_state().
      This will be set via transform_config_dict().
      To get the state from another recurrent layer, use the GetLastHiddenStateLayer (get_last_hidden_state).
    :param None initial_output: the initial output is defined implicitly via initial state, thus don't set this
    """
    super(RnnCellLayer, self).__init__(n_out=n_out, **kwargs)
    assert self._rec_previous_layer or self.input_data.time_dim_axis is not None, (
      "%s: This layer is expected to be used inside a RecLayer, or to have input with time." % self)
    self._initial_state = initial_state
    assert initial_output is None, "set initial_state instead"
    import re
    from TFUtil import get_initializer
    # Cannot use self.var_creation_scope() when this is inside a RecLayer.
    with reuse_name_scope(
      "rec",
      initializer=get_initializer(
        weights_init, seed=self.network.random.randint(2 ** 31), eval_local_ns={"layer": self}),
      reuse=getattr(tf, "AUTO_REUSE", None)
    ) as scope:
      assert isinstance(scope, tf.VariableScope)
      scope_name_prefix = scope.name + "/"  # e.g. "layer1/rec/"
      self.cell = self._get_cell(n_out=n_out, unit=unit, unit_opts=unit_opts)
      assert isinstance(self.cell, rnn_cell.RNNCell)
      if self._rec_previous_layer:
        x = self.input_data.placeholder
        if isinstance(self.cell, BaseRNNCell):
          x = self.cell.get_input_transformed(x)
        assert not self.input_data or self.input_data.time_dim_axis is None
        self.output.time_dim_axis = None
        self.output.batch_dim_axis = 0
        prev_state = self._rec_previous_layer.rec_vars_outputs["state"]
        self.output.placeholder, state = self.cell(x, prev_state)
      else:
        assert self.input_data and self.input_data.time_dim_axis is not None
        x = self.input_data.get_placeholder_as_time_major()
        if isinstance(self.cell, BaseRNNCell):
          x = self.cell.get_input_transformed(x)
        self.output.time_dim_axis = 0
        self.output.batch_dim_axis = 1
        state0 = self.get_rec_initial_state(
          n_out=n_out, unit=unit, unit_opts=unit_opts,
          batch_dim=self.input_data.get_batch_dim(), name=self.name,
          initial_state=initial_state)
        self.output.placeholder, state = tf.nn.dynamic_rnn(
          self.cell,
          inputs=x,
          sequence_length=self.input_data.get_sequence_lengths(),
          initial_state=state0, time_major=True, scope=scope)
      self._hidden_state = state
      self.rec_vars_outputs["state"] = state
      params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=re.escape(scope_name_prefix))
      assert params
      self.params.update({p.name[len(scope_name_prefix):-2]: p for p in params})

  @classmethod
  def _get_cell(cls, n_out, unit, unit_opts=None):
    """
    :param int n_out:
    :param str|rnn_cell.RNNCell unit:
    :param dict[str]|None unit_opts:
    :rtype: rnn_cell.RNNCell|TFNativeOp.RecSeqCellOp
    """
    if isinstance(unit, rnn_cell.RNNCell):
      return unit
    rnn_cell_class = RecLayer.get_rnn_cell_class(unit)
    # E.g. rnn_cell_class is :class:`rnn_cell.LSTMCell`.
    if issubclass(rnn_cell_class, rnn_cell.RNNCell):
      if unit_opts is None:
        unit_opts = {}
      assert isinstance(unit_opts, dict)
      # This should not have any side-effects, i.e. it should not add to the current computation graph,
      # it should also not create any vars yet, etc.
      cell = rnn_cell_class(n_out, **unit_opts)
      assert isinstance(cell, rnn_cell.RNNCell)
      return cell
    import TFNativeOp
    if issubclass(rnn_cell_class, TFNativeOp.RecSeqCellOp):
      return rnn_cell_class(n_hidden=n_out)
    raise TypeError("does not expect %r here for unit %r" % (rnn_cell_class, unit))

  @classmethod
  def get_out_data_from_opts(cls, n_out, name, sources=(), **kwargs):
    """
    :param int n_out:
    :param str name: layer name
    :param list[LayerBase] sources:
    :rtype: Data
    """
    beam_size = None
    for dep in sources:
      beam_size = beam_size or dep.output.beam_size
    shape = (n_out,)
    batch_dim_axis = 0
    time_dim_axis = None
    if sources and sources[0].output.time_dim_axis is not None:
      shape = (None,) + shape
      batch_dim_axis = 1
      time_dim_axis = 0
    return Data(
      name="%s_output" % name,
      shape=shape, dim=n_out,
      batch_dim_axis=batch_dim_axis,
      time_dim_axis=time_dim_axis,
      size_placeholder={} if not sources else sources[0].output.size_placeholder.copy(),
      beam_size=beam_size)

  def get_dep_layers(self):
    l = list(super(RnnCellLayer, self).get_dep_layers())

    def visit(s):
      if isinstance(s, (list, tuple)):
        for x in s:
          visit(x)
      elif isinstance(s, dict):
        for x in s.values():
          visit(x)
      elif isinstance(s, LayerBase):
        l.append(s)
      else:
        assert isinstance(s, (str, int, float, type(None)))

    visit(self._initial_state)
    return l

  @classmethod
  def get_hidden_state_size(cls, n_out, unit, unit_opts=None, **kwargs):
    """
    :param int n_out:
    :param str unit:
    :param dict[str]|None unit_opts:
    :return: size or tuple of sizes
    :rtype: int|tuple[int]
    """
    cell = cls._get_cell(unit=unit, unit_opts=unit_opts, n_out=n_out)
    return cell.state_size

  @classmethod
  def get_output_from_state(cls, state, unit):
    """
    :param tuple[tf.Tensor]|tf.Tensor state:
    :param str unit:
    :rtype: tf.Tensor
    """
    import tensorflow.contrib.rnn as rnn_contrib
    if isinstance(state, rnn_contrib.LSTMStateTuple):
      return state.h
    # Assume the state is the output. This might be wrong...
    assert isinstance(state, tf.Tensor)
    return state

  def get_hidden_state(self):
    """
    :return: state as defined by the cell
    :rtype: tuple[tf.Tensor]|tf.Tensor
    """
    return self._hidden_state

  @classmethod
  def get_state_by_key(cls, state, key):
    """
    :param tf.Tensor|tuple[tf.Tensor]|namedtuple state:
    :param int|str|None key:
    :rtype: tf.Tensor
    """
    from tensorflow.python.util import nest
    from Util import is_namedtuple
    if key == "*":
      if nest.is_sequence(state):
        x = tf.concat(state, axis=-1)  # in dim-axis
      else:
        x = state
    elif key == "flat":
      assert nest.is_sequence(state), "only a sequence can be flattened, but got %r" % (state,)
      x = tf.concat(state, axis=-1)  # in dim-axis
    elif is_namedtuple(type(state)):
      assert isinstance(key, str), "state %r is a named tuple, thus key %r must be a string" % (state, key)
      x = getattr(state, key)
    elif nest.is_sequence(state):
      assert isinstance(key, int), "state %r is a tuple, thus key %r must be an int" % (state, key)
      x = state[key]
    else:
      assert isinstance(state, tf.Tensor), "unexpected state %r" % (state,)
      assert key is None, "state %r is a tensor, thus key %r must be None" % (state, key)
      x = state
    assert isinstance(x, tf.Tensor)
    x.set_shape(tf.TensorShape([None, None]))  # (batch,dim)
    return x

  def get_last_hidden_state(self, key):
    return self.get_state_by_key(self._hidden_state, key=key)

  @classmethod
  def get_rec_initial_state(cls, batch_dim, name, n_out, unit, initial_state=None, unit_opts=None,
                            rec_layer=None, **kwargs):
    """
    Very similar to :func:`get_rec_initial_output`.
    Initial hidden state when used inside a recurrent layer for the frame t=-1, if it is needed.
    As arguments, we get the usual layer arguments.
    batch_dim is added because it might be special because of beam search.
    Also see :func:`transform_config_dict` for `initial_state`.

    Note: This could maybe share code with :func:`get_rec_initial_output`,
    although it is a bit more generic here because the state can also be a namedtuple
    or any kind of nested structure.

    :param tf.Tensor batch_dim: including beam size in beam search
    :param str name: layer name
    :param int n_out: out dim
    :param str unit: cell name
    :param dict[str]|None unit_opts:
    :param LayerBase|str|int|float|None|list|tuple|namedtuple initial_state: see code
    :param RecLayer|None rec_layer: for the scope
    :rtype: tf.Tensor|tuple[tf.Tensor]|namedtuple
    """
    from Util import dummy_noop_ctx
    with tf.name_scope("rec_initial_state"):
      init_value = initial_state
      dim = cls.get_hidden_state_size(n_out=n_out, unit=unit, unit_opts=unit_opts, **kwargs)

      def make(d, v, key):
        """
        :param int d: dim
        :param LayerBase|int|float|str|None v: value
        :param str|int|None key:
        :return: tensor/variable for the state
        :rtype: tf.Tensor
        """
        assert isinstance(d, int)
        assert isinstance(v, (LayerBase, int, float, str, type(None)))
        assert isinstance(key, (str, int, type(None)))
        key_name = str(key if key is not None else "var")
        shape = [batch_dim, d]
        if isinstance(v, LayerBase):
          h = v.get_last_hidden_state(key="*")
          if h is not None:
            h.set_shape(tf.TensorShape((None, d)))
            return h
          assert v.output.batch_dim_axis == 0
          assert v.output.time_dim_axis is None
          assert v.output.shape == (d,)
          return v.output.placeholder
        elif v == "zeros" or not v:
          return tf.zeros(shape)
        elif v == "ones" or v == 1:
          return tf.ones(shape)
        elif v == "var":
          with rec_layer.var_creation_scope() if rec_layer else dummy_noop_ctx():
            var = tf.get_variable("initial_%s" % key_name, shape=(d,), initializer=tf.zeros_initializer())
          from TFUtil import expand_dims_unbroadcast
          var = expand_dims_unbroadcast(var, axis=0, dim=batch_dim)  # (batch,dim)
          return var
        elif v == "keep_over_epoch":
          assert rec_layer is not None
          with rec_layer.var_creation_scope():
            var = tf.get_variable(
              'keep_state_%s' % key_name,
              validate_shape=False, initializer=tf.zeros(()),  # dummy state, will not be used like this
              trainable=False)
          assert isinstance(var, tf.Variable)
          var.set_shape((None, d))
          rec_layer.saveable_param_replace[var] = None  # Do not save this variable.

          def update_var():
            with tf.control_dependencies([
                  tf.assign(var, rec_layer.get_last_hidden_state(key=key), validate_shape=False)]):
              rec_layer.output.placeholder = tf.identity(rec_layer.output.placeholder)
          rec_layer.post_init_hooks.append(update_var)
          step = rec_layer.network.get_epoch_step()
          s = tf.cond(tf.equal(step, 0), lambda: tf.zeros(shape), lambda: var.value())
          s.set_shape((None, d))
          return s
        else:
          raise Exception("invalid initial state type %r for sub-layer %r, key %r" % (v, name, key))

      def make_list(keys):
        assert isinstance(keys, (tuple, list))
        assert len(keys) == len(dim)
        if isinstance(init_value, (list, tuple)):
          assert len(init_value) == len(dim)
          return [make(d, v_, k) for (d, v_, k) in zip(dim, init_value, keys)]
        # Do not broadcast LayerBase automatically in this case.
        assert isinstance(init_value, (int, float, str, type(None)))
        return [make(d, init_value, k) for d, k in zip(dim, keys)]

      # Make it the same type because nest.assert_same_structure() will complain otherwise.
      from Util import is_namedtuple
      if is_namedtuple(type(dim)):
        keys = dim._fields
        assert len(dim) == len(keys)
        assert isinstance(init_value, (int, float, str, tuple, list, dict, type(None)))
        if not isinstance(init_value, dict) and init_value not in (0, 1, None) and not isinstance(init_value, str):
          print("Layer %r: It is recommended to use a dict to specify 'initial_state' with keys %r for the state dimensions %r." % (name, keys, dim), file=log.v2)
        if isinstance(init_value, dict):
          assert set(init_value.keys()) == set(keys), "You must specify all keys for the state dimensions %r." % dim
          assert len(init_value) == len(dim)
          s = {k: make(d, init_value[k], key=k) for (k, d) in zip(keys, dim)}
        else:
          s = make_list(keys=keys)
          assert len(s) == len(keys)
          s = {k: s_ for (k, s_) in zip(keys, s)}
        return type(dim)(**s)
      elif isinstance(dim, (tuple, list)):
        s = make_list(keys=[i for i in range(len(dim))])
        assert len(s) == len(dim)
        return type(dim)(s)
      elif isinstance(dim, int):
        return make(dim, init_value, key=None)
      else:
        raise Exception("Did not expect hidden_state_size %r." % dim)

  @classmethod
  def get_rec_initial_extra_outputs(cls, **kwargs):
    return {"state": cls.get_rec_initial_state(**kwargs)}

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param TFNetwork.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    super(RnnCellLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if "initial_state" in d:
      d["initial_state"] = cls.transform_initial_state(d["initial_state"], network=network, get_layer=get_layer)

  @staticmethod
  def transform_initial_state(initial_state, network, get_layer):
    """
    :param str|float|int|list[str|float|int]|dict[str]|None initial_state:
    :param TFNetwork.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    def resolve(v):
      if isinstance(v, str):
        if v in ["zeros", "ones", "var", "keep_over_epoch"]:
          return v
        return get_layer(v)
      if isinstance(v, (tuple, list)):
        return [resolve(x) for x in v]
      if isinstance(v, dict):
        return {k: resolve(x) for (k, x) in v.items()}
      if isinstance(v, (float, int)):
        return v
      if v is None:
        return v
      raise Exception("initial_state %r: invalid type: %r, %r" % (initial_state, v, type(v)))
    return resolve(initial_state)

  @classmethod
  def get_rec_initial_output(cls, unit, initial_output=None, initial_state=None, **kwargs):
    assert initial_output is None, "layer %r: use initial_state instead" % kwargs["name"]
    if initial_state in [None, 0, "zeros"]:
      # We can just return 0.
      return super(RnnCellLayer, cls).get_rec_initial_output(initial_output=0, **kwargs)
    state = cls.get_rec_initial_state(unit=unit, initial_state=initial_state, **kwargs)
    return cls.get_output_from_state(state=state, unit=unit)


class GetLastHiddenStateLayer(LayerBase):
  """
  Will combine (concat or add or so) all the last hidden states from all sources.
  """

  layer_class = "get_last_hidden_state"

  def __init__(self, n_out, combine="concat", key='*', **kwargs):
    """
    :param int n_out: dimension. output will be of shape (batch, n_out)
    :param str combine: "concat" or "add"
    :param str|int|None key: for the state, which could be a namedtuple. see :func:`RnnCellLayer.get_state_by_key`
    """
    super(GetLastHiddenStateLayer, self).__init__(**kwargs)
    assert len(self.sources) > 0
    last_states = [s.get_last_hidden_state(key=key) for s in self.sources]
    assert all([s is not None for s in last_states])
    if len(last_states) == 1:
      h = last_states[0]
    else:
      if combine == "concat":
        h = tf.concat(last_states, axis=-1, name="concat_hidden_states")
      elif combine == "add":
        h = tf.add_n(last_states, name="add_hidden_states")
      else:
        raise Exception("invalid hidden states combine mode %r" % combine)
    from TFUtil import check_input_ndim, check_input_dim
    h = check_input_ndim(h, 2)
    h = check_input_dim(h, 1, n_out)
    self.output.placeholder = h

  def get_last_hidden_state(self, key):
    assert key in [None, '*']
    return self.output.placeholder

  @classmethod
  def get_out_data_from_opts(cls, n_out, **kwargs):
    return super(GetLastHiddenStateLayer, cls).get_out_data_from_opts(
      out_type={"shape": (n_out,), "dim": n_out, "batch_dim_axis": 0, "time_dim_axis": None}, **kwargs)


class GetRecAccumulatedOutput(LayerBase):
  """
  Retrieves the accumulated output from a Subnet-layer.
  Note: the source-layer must be marked as output layer
        with "is_output_layer": True
  """

  layer_class = "get_rec_accumulated"

  def __init__(self, sub_layer, **kwargs):
    """
    :param LayerBase sub_layer: sub-layer to get the outputs from
    """
    super(GetRecAccumulatedOutput, self).__init__(**kwargs)
    assert len(self.sources) == 1
    assert sub_layer is not None
    rec_layer = self.sources[0]  # type: RecLayer
    cell = rec_layer.cell  # type: _SubnetworkRecCell
    self.output.placeholder = cell.final_acc_tas_dict["output_%s" % sub_layer.name].stack()
    self.output.size_placeholder = {
      axis + 1: placeholder for axis, placeholder in sub_layer.output.size_placeholder.items()
    }
    self.output.size_placeholder[0] = rec_layer.output.size_placeholder[0]

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param TFNetwork.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    assert "from" in d
    sources = d.pop("from", [])
    sub_layer = d.pop("sub_layer", None)
    assert sub_layer is not None, "Set sub_layer"
    if isinstance(sources, str):
      sources = [sources]
    assert len(sources) == 1
    assert isinstance(sources[0], str)
    rec_layer = get_layer(sources[0])
    assert isinstance(rec_layer.cell, _SubnetworkRecCell)
    template = rec_layer.cell.layer_data_templates[sub_layer]
    d["sub_layer"] = template
    d["sources"] = [rec_layer]

  @classmethod
  def get_out_data_from_opts(cls, sources, sub_layer, **kwargs):
    """
    :param list[LayerBase] sources:
    :param LayerBase sub_layer:
    :rtype: Data
    """
    assert len(sources) == 1
    return Data(
      name="%s_output" % kwargs["name"],
      shape=(None,) + sub_layer.output.shape,
      dtype=sub_layer.output.dtype,
      time_dim_axis=0,
      batch_dim_axis=sub_layer.output.batch_dim_axis + 1)


class ChoiceLayer(LayerBase):
  """
  This layer represents a choice to be made in search during inference,
  such as choosing the top-k outputs from a log-softmax for beam search.
  During training, this layer can return the true label.
  This is supposed to be used inside the rec layer.
  This can be extended in various ways.

  We present the scores in +log space, and we will add them up along the path.
  Assume that we get input (batch,dim) from a (log-)softmax.
  Assume that each batch is already a choice via search.
  In search with a beam size of N, we would output
  sparse (batch=N,) and scores for each.
  """
  layer_class = "choice"

  _debug_out = None  # type: None|list

  def __init__(self, beam_size, input_type="prob", explicit_search_source=None, length_normalization=True,
               scheduled_sampling=False,
               cheating=False,
               **kwargs):
    """
    :param int beam_size: the outgoing beam size. i.e. our output will be (batch * beam_size, ...)
    :param str input_type: "prob" or "log_prob", whether the input is in probability space, log-space, etc.
      or "regression", if it is a prediction of the data as-is.
    :param LayerBase|None explicit_search_source: will mark it as an additional dependency
    :param dict|None scheduled_sampling:
    :param bool length_normalization: evaluates score_t/len in search
    :param bool cheating: if True, will always add the true target in the beam
    """
    super(ChoiceLayer, self).__init__(**kwargs)
    from Util import CollectionReadCheckCovered
    self.explicit_search_source = explicit_search_source
    self.scheduled_sampling = CollectionReadCheckCovered.from_bool_or_dict(scheduled_sampling)
    # We assume log-softmax here, inside the rec layer.
    assert self.target
    if self.network.search_flag:
      assert len(self.sources) == 1
      assert not self.sources[0].output.sparse
      assert self.sources[0].output.dim == self.output.dim
      assert self.sources[0].output.shape == (self.output.dim,)
      # We are doing the search.
      self.search_choices = SearchChoices(
        owner=self,
        beam_size=beam_size)
      if input_type == "regression":
        # It's not a probability distribution, so there is no search here.
        net_batch_dim = self.network.get_data_batch_dim()
        assert self.search_choices.beam_size == 1
        assert not cheating
        self.output = self.sources[0].output.copy_compatible_to(self.output)
        self.search_choices.src_beams = tf.zeros((net_batch_dim, 1), dtype=tf.int32)
        self.search_choices.set_beam_scores(self.search_choices.src_layer.search_choices.beam_scores)
      else:
        net_batch_dim = self.network.get_data_batch_dim()
        assert self.search_choices.src_layer, (
          self.network.debug_search_choices(base_search_choice=self),
          "Not implemented yet. In rec-layer, we would always have our prev-frame as one previous search choice. "
          "Our deps: %r" % self.get_dep_layers())
        scores_base = self.search_choices.src_layer.search_choices.beam_scores  # (batch, beam_in)
        assert scores_base.get_shape().ndims == 2, "%r invalid" % self.search_choices.src_layer.search_choices
        base_beam_in = tf.shape(scores_base)[1]  # 1 in first frame, then beam_in (beam_size)
        scores_beam_in = tf.shape(self.sources[0].output.placeholder)[0] // net_batch_dim
        beam_size = self.sources[0].output.beam_size
        # About incoming beam size:
        #   base_beam_in  - 1 in first frame, then beam_in
        #   scores_beam_in  - beam_size or 1
        #   beam_size  - beam_in
        # Note about scores_beam_in, i.e. the batch-beam-size of other layers:
        # We could make it like base_beam_in, i.e. have beam-size 1 in the 0th layer
        # and also in the 1st layer before any ChoiceLayer.
        # However, currently it makes the code a bit simpler to just have always
        # the final beam-size everywhere.
        # Keep in mind that this might change at some future point.
        if length_normalization:
          assert self.network.have_rec_step_info()
          t = self.network.get_rec_step_index()  # scalar
          end_flags_flat = self.network.get_rec_step_info().get_end_flag()  # (batch * beam_in,)
          with tf.name_scope("length_normalization"):
            end_flags = tf.reshape(end_flags_flat, [net_batch_dim, beam_size])  # (batch, beam_in)
            end_flags = end_flags[:, :base_beam_in]  # see scores_in below
            # Normalized scores, so we evaluate score_t/len.
            # If seq ended, score_t/t == score_{t-1}/(t-1), thus score_t = score_{t-1}*(t/(t-1))
            # Because we count with EOS symbol, shifted by one.
            scores_base *= tf.where(
              end_flags,
              tf.ones(tf.shape(end_flags)) * (tf.to_float(t + 1) / tf.to_float(t)),
              tf.ones(tf.shape(end_flags)))
        scores_base = tf.expand_dims(scores_base, axis=-1)  # (batch, beam_in, dim)
        scores_in = self.sources[0].output.placeholder  # (batch * beam_in, dim)
        # We present the scores in +log space, and we will add them up along the path.
        if input_type == "prob":
          if self.sources[0].output_before_activation:
            scores_in = self.sources[0].output_before_activation.get_log_output()
          else:
            from TFUtil import safe_log
            scores_in = safe_log(scores_in)
        elif input_type == "log_prob":
          pass
        else:
          raise Exception("%r: invalid input type %r" % (self, input_type))
        from TFUtil import filter_ended_scores
        scores_in_dim = self.sources[0].output.dim
        if self.network.have_rec_step_info():
          scores_in = filter_ended_scores(
            scores_in, end_flags=self.network.get_rec_step_info().get_end_flag(),
            dim=scores_in_dim, batch_dim=net_batch_dim * scores_beam_in)  # (batch * beam_in, dim)
        scores_in = tf.reshape(scores_in, [net_batch_dim, scores_beam_in, scores_in_dim])  # (batch, beam_in, dim)
        with tf.control_dependencies([
              # See comment above. This checks that all is as expected.
              tf.Assert(tf.logical_or(
                tf.equal(base_beam_in, 1),
                tf.logical_and(
                  tf.equal(base_beam_in, scores_beam_in),
                  tf.equal(base_beam_in, beam_size))),
                [
                  "base_beam_in", base_beam_in,
                  "scores_beam_in", scores_beam_in,
                  "beam_size", beam_size])]):
          # See the comment above. It could be that scores_in has a wider beam
          # than what should be used here now.
          scores_in = scores_in[:, :base_beam_in]  # (batch, beam_in, dim)
        scores_comb = scores_in + scores_base  # (batch, beam_in, dim)
        scores_comb_flat = tf.reshape(
          scores_comb, [net_batch_dim, base_beam_in * scores_in_dim])  # (batch, beam_in * dim)
        # `tf.nn.top_k` is the core function performing our search.
        # We get scores/labels of shape (batch, beam) with indices in [0..beam_in*dim-1].
        scores, labels = tf.nn.top_k(scores_comb_flat, k=beam_size)
        if cheating:
          # It assumes that sorted=True in top_k, and the last entries in scores/labels are the worst.
          # We replace them by the true labels.
          gold_targets = self._static_get_target_value(
            target=self.target, network=self.network,
            mark_data_key_as_used=True).get_placeholder_as_batch_major()  # (batch*beam,), int32
          # gold_targets will get automatically expanded for the beam. Undo that.
          gold_targets = tf.reshape(gold_targets, [net_batch_dim, beam_size])[:, 0]
          gold_beam_in_idx = base_beam_in - 1  # also assume last index
          gold_labels = gold_beam_in_idx * scores_in_dim + gold_targets  # (batch,)
          gold_labels_bc = tf.expand_dims(gold_labels, axis=1)  # (batch,1)
          labels = tf.concat([labels[:, :beam_size - 1], gold_labels_bc], axis=1)  # (batch,beam)
          from TFUtil import nd_indices
          gold_scores = tf.gather_nd(
            scores_comb[:, gold_beam_in_idx], indices=nd_indices(gold_targets))  # (batch,)
          gold_scores_bc = tf.expand_dims(gold_scores, axis=1)  # (batch,1)
          scores = tf.concat([scores[:, :beam_size - 1], gold_scores_bc], axis=1)  # (batch,beam)
        self.search_choices.src_beams = labels // scores_in_dim  # (batch, beam) -> beam_in idx
        labels = labels % scores_in_dim  # (batch, beam) -> dim idx
        labels = tf.reshape(labels, [net_batch_dim * beam_size])  # (batch * beam)
        labels = tf.cast(labels, self.output.dtype)
        self.search_choices.set_beam_scores(scores)  # (batch, beam) -> log score
        if self._debug_out is not None:
          from TFUtil import identity_with_debug_log
          labels = identity_with_debug_log(
            out=self._debug_out, x=labels, args={
              "step": self.network.get_rec_step_index() if self.network.have_rec_step_info() else tf.constant(-1),
              "base_beam_in": base_beam_in,
              "scores_in_orig": self.sources[0].output.placeholder,
              "scores_in": scores_in,
              "scores_base_orig": self.search_choices.src_layer.search_choices.beam_scores,
              "scores_base": scores_base,
              "scores_combined": scores_comb,
              "src_beam_idxs": self.search_choices.src_beams,
              "labels": tf.reshape(labels, [net_batch_dim, beam_size]),
              "scores": scores})
        self.output = Data(
          name="%s_choice_output" % self.name,
          batch_dim_axis=0,
          shape=self.output.shape,
          sparse=True,
          dim=self.output.dim,
          dtype=self.output.dtype,
          placeholder=labels,
          available_for_inference=True,
          beam_size=beam_size)
    elif self.scheduled_sampling.truth_value:
      # Original paper: https://arxiv.org/abs/1506.03099
      # Currently, here: no scheduling, just always sample...
      # We could also do that with a beam (num_samples=beam_size). But currently we do not.
      # Note that in other implementations (e.g. tensor2tensor), as well as in the original paper,
      # they do the sampling from the logits where the decoder got always the true labels,
      # and then a second pass is done for the decoder, and the losses are used only from the second pass.
      # This means that they don't back-propagate the gradient of the losses through the sampling decision,
      # as they write in the paper.
      # This is different from what we do here. There is no second pass.
      # Currently there is also no gradient through tf.multinomial but we could add that later.
      assert len(self.sources) == 1
      scores_in = self.sources[0].output.get_placeholder_as_batch_major()  # (batch, dim)
      # We present the scores in +log space, and we will add them up along the path.
      if input_type == "prob":
        scores_in = tf.log(scores_in)
      elif input_type == "log_prob":
        pass
      else:
        raise Exception("%r: invalid input type %r" % (self, input_type))
      samples = tf.multinomial(scores_in, num_samples=1)  # (batch, num_samples), int64
      samples = tf.to_int32(tf.reshape(samples, [-1]))  # (batch,), int32
      if self.scheduled_sampling.get("gold_mixin_prob"):
        gold_targets = self._static_get_target_value(
          target=self.target, network=self.network,
          mark_data_key_as_used=True).get_placeholder_as_batch_major()  # (batch,), int32
        samples = tf.where(
          tf.less(tf.random_uniform(tf.shape(samples)), self.scheduled_sampling.get("gold_mixin_prob")),
          gold_targets, samples)
      self.output = Data(
        name="%s_sampled_output" % self.name,
        batch_dim_axis=0,
        shape=self.output.shape,
        sparse=True,
        dim=self.output.dim,
        dtype=self.output.dtype,
        placeholder=samples,
        available_for_inference=True)
    else:  # no search, and no scheduled-sampling
      assert len(self.sources) == 0  # will be filtered out in transform_config_dict
      # Note: If you want to do forwarding, without having the reference,
      # that wont work. You must do search in that case.
      self.output = self._static_get_target_value(
        target=self.target, network=self.network,
        mark_data_key_as_used=True).copy()
      self.output.available_for_inference = True  # in inference, we should do search

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param TFNetwork.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    if not network.search_flag and not d.get("scheduled_sampling"):
      # In the dependency graph, we don't want it.
      # This can enable some optimizations in the RecLayer.
      # We do it here because we should know about the deps early in the template creation in RecLayer.
      d["from"] = []
    if d.get("explicit_search_source"):
      d["explicit_search_source"] = get_layer(d["explicit_search_source"]) if network.search_flag else None
    super(ChoiceLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

  @classmethod
  def get_out_data_from_opts(cls, target, network, beam_size, **kwargs):
    out = cls._static_get_target_value(
      target=target, network=network,
      mark_data_key_as_used=False).copy()
    out.available_for_inference = True  # in inference, we would do search
    if network.search_flag:
      out.beam_size = beam_size
    return out

  # noinspection PyMethodOverriding
  @classmethod
  def get_rec_initial_extra_outputs(cls, network, beam_size, **kwargs):
    """
    :param TFNetwork.TFNetwork network:
    :param int beam_size:
    :rtype: dict[str,tf.Tensor]
    """
    if not network.search_flag:
      return {}
    batch_dim = network.get_data_batch_dim()
    # Note: Use beam_size 1 for the initial as there are no competing hypotheses yet.
    initial_scores = tf.zeros([batch_dim, 1])  # (batch, beam)
    return {"choice_scores": initial_scores}

  @classmethod
  def get_rec_initial_extra_outputs_shape_invariants(cls, **kwargs):
    # Initial beam size is 1 and then later the given one, so it changes.
    return {"choice_scores": tf.TensorShape((None, None))}  # (batch, beam)

  def get_dep_layers(self):
    # See also self.transform_config_dict where we might strip away the sources.
    l = super(ChoiceLayer, self).get_dep_layers()
    if self.explicit_search_source:
      l.append(self.explicit_search_source)
    return l


class DecideLayer(LayerBase):
  """
  This is kind of the counter-part to the choice layer.
  This only has an effect in search mode.
  E.g. assume that the input is of shape (batch * beam, time, dim)
  and has search_sources set.
  Then this will output (batch, time, dim) where the beam with the highest score is selected.
  Thus, this will do a decision based on the scores.
  In will convert the data to batch-major mode.
  """
  layer_class = "decide"

  def __init__(self, length_normalization=False, **kwargs):
    """
    :param bool length_normalization: performed on the beam scores
    """
    super(DecideLayer, self).__init__(**kwargs)
    # If not in search, this will already be set via self.get_out_data_from_opts().
    if self.network.search_flag:
      assert len(self.sources) == 1
      src = self.sources[0]
      self.decide(src=src, output=self.output, length_normalization=length_normalization)
      self.search_choices = SearchChoices(owner=self, is_decided=True)

  @classmethod
  def decide(cls, src, output=None, name=None, length_normalization=False):
    """
    :param LayerBase src: with search_choices set. e.g. input of shape (batch * beam, time, dim)
    :param Data|None output:
    :param str|None name:
    :param bool length_normalization: performed on the beam scores
    :return: best beam selected from input, e.g. shape (batch, time, dim)
    :rtype: Data
    """
    search_choices = src.get_search_choices()
    assert search_choices
    if not output:
      output = src.output.copy_template(name="%s_output" % (name or src.name)).copy_as_batch_major()
    assert output.batch_dim_axis == 0
    batch_dim = src.network.get_data_batch_dim()
    src_data = src.output.copy_as_batch_major()
    beam_size = search_choices.beam_size
    src_output = tf.reshape(
      src_data.placeholder,
      [batch_dim, beam_size] +
      [tf.shape(src_data.placeholder)[i] for i in range(1, src_data.batch_ndim)])  # (batch, beam, [time], [dim])
    # beam_scores is of shape (batch, beam) -> +log score.
    beam_scores = search_choices.beam_scores
    if length_normalization:
      beam_scores /= tf.to_float(tf.reshape(src.output.get_sequence_lengths(), [batch_dim, beam_size]))
    beam_idxs = tf.argmax(beam_scores, axis=1)  # (batch,)
    from TFUtil import assert_min_tf_version, nd_indices
    assert_min_tf_version((1, 1), "gather_nd")
    beam_idxs_ext = nd_indices(beam_idxs)
    output.placeholder = tf.cond(
      tf.greater(tf.size(src_output), 0),  # can happen to be empty
      lambda: tf.gather_nd(src_output, indices=beam_idxs_ext),
      lambda: src_output[:, 0], name="cond_not_empty")  # (batch, [time], [dim])
    output.size_placeholder = {}
    for i, size in src_data.size_placeholder.items():
      size = tf.reshape(size, [batch_dim, beam_size])  # (batch, beam)
      output.size_placeholder[i] = tf.gather_nd(size, indices=beam_idxs_ext)  # (batch,)
    return output

  @classmethod
  def get_out_data_from_opts(cls, name, sources, network, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param TFNetwork.TFNetwork network:
    :rtype: Data
    """
    assert len(sources) == 1
    if network.search_flag:
      data = sources[0].output.copy_template(name="%s_output" % name).copy_as_batch_major()
      data.beam_size = None
      return data
    else:
      return sources[0].output


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
    self.base_weights = None  # type: None|tf.Tensor  # (batch, base_time), see self.get_base_weights()

  def get_dep_layers(self):
    return super(AttentionBaseLayer, self).get_dep_layers() + [self.base]

  def get_base_weights(self):
    """
    We can formulate most attentions as some weighted sum over the base time-axis.

    :return: the weighting of shape (batch, base_time), in case it is defined
    :rtype: tf.Tensor|None
    """
    return self.base_weights

  def get_base_weight_last_frame(self):
    """
    From the base weights (see self.get_base_weights(), must return not None)
    takes the weighting of the last frame in the time-axis (according to sequence lengths).

    :return: shape (batch,) -> float (number 0..1)
    :rtype: tf.Tensor
    """
    last_frame_idxs = tf.maximum(self.base.output.get_sequence_lengths() - 1, 0)  # (batch,)
    from TFUtil import assert_min_tf_version, nd_indices
    assert_min_tf_version((1, 1), "gather_nd")
    last_frame_idxs_ext = nd_indices(last_frame_idxs)
    return tf.gather_nd(self.get_base_weights(), indices=last_frame_idxs_ext)  # (batch,)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    super(AttentionBaseLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["base"] = get_layer(d["base"])

  @classmethod
  def get_out_data_from_opts(cls, name, base, n_out=None, **kwargs):
    """
    :param str name:
    :param int|None n_out:
    :param LayerBase base:
    :rtype: Data
    """
    out = base.output.copy_template_excluding_time_dim().copy(name="%s_output" % name)
    assert out.time_dim_axis is None
    assert out.batch_dim_axis == 0
    if n_out:
      assert out.dim == n_out, (
        "The default attention selects some frame-weighted input of shape [batch, frame, dim=%i]," % out.dim +
        " thus resulting in [batch, dim=%i] but you specified n_out=%i." % (out.dim, n_out))
    return out


class GlobalAttentionContextBaseLayer(AttentionBaseLayer):
  def __init__(self, base_ctx, **kwargs):
    """
    :param LayerBase base: encoder output to attend on
    :param LayerBase base_ctx: encoder output used to calculate the attention weights
    """
    super(GlobalAttentionContextBaseLayer, self).__init__(**kwargs)
    self.base_ctx = base_ctx

  def get_dep_layers(self):
    return super(GlobalAttentionContextBaseLayer, self).get_dep_layers() + [self.base_ctx]

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    super(GlobalAttentionContextBaseLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["base_ctx"] = get_layer(d["base_ctx"])


class GenericAttentionLayer(AttentionBaseLayer):
  """
  The weighting for the base is specified explicitly here.
  This can e.g. be used together with :class:`SoftmaxOverSpatialLayer`.
  """
  layer_class = "generic_attention"

  def __init__(self, weights, auto_squeeze=True, **kwargs):
    """
    :param LayerBase base: encoder output to attend on. (B, enc-time)|(enc-time, B) + (...) + (n_out,)
    :param LayerBase weights: attention weights. ((B, enc-time)|(enc-time, B)) + (1,)|()
    :param bool auto_squeeze: auto-squeeze any weight-axes with dim=1 away
    """
    super(GenericAttentionLayer, self).__init__(**kwargs)
    assert not self.sources, "only base and weights are needed"
    self.weights = weights
    weights_data = self.weights.output.copy_as_batch_major()
    self.base_weights = weights_data.copy_with_time_dim_axis(1)  # (B,T,...)
    # We will always have batch-major mode and just write T for enc-time, i.e. (B,T,...).
    weights_t = weights_data.placeholder  # (B,...,T,...)
    # Move time-dim to end in weights.
    new_axes = list(range(weights_data.batch_ndim))
    new_axes.remove(weights_data.time_dim_axis)
    new_axes.append(weights_data.time_dim_axis)
    weights_t = tf.transpose(weights_t, perm=new_axes)  # (B,...,T)
    shape = tf.shape(weights_t)
    shape = [weights_data.batch_shape[new_axes[i]] or shape[i] for i in range(len(new_axes))]
    time_dim = shape[-1]
    # rem_dims_start_axis will specify which prefix axes we will share,
    # and thus also in the output we will have shape[:rem_dims_start_axis].
    # Example: weights (B,1,T), base (B,T,V) -> rem_dims_start_axis=1, rem_dims=[]
    # Example: weights (B,H,1,T), base (B,T,H,V) -> rem_dims_start_axis=2, rem_dims=[] (auto_squeeze)
    # Example: weights (B,H,T), base (B,T,H,V) -> rem_dims_start_axis=2, rem_dims=[]
    if weights_data.batch_ndim < self.base.output.batch_ndim:  # weights_t of shape (B,T)
      from TFUtil import expand_multiple_dims
      weights_t = expand_multiple_dims(
        weights_t, [-2] * (self.base.output.batch_ndim - weights_data.batch_ndim))  # (B,...,1,T)
      rem_dims_start_axis = self.base.output.batch_ndim - 2  # count without T,V
      rem_dims = []
    else:
      rem_dims_start_axis = self.base.output.batch_ndim - 2  # count without T,V
      rem_dims = shape[rem_dims_start_axis:-1]
      weights_t = tf.reshape(
        weights_t, shape[:rem_dims_start_axis] + [tf.reduce_prod(rem_dims) if rem_dims else 1, time_dim])  # (B,?,T)
    if auto_squeeze:
      rem_dims = [d for d in rem_dims if d != 1]
    base = self.base.output.copy_as_batch_major().copy_with_time_dim_axis(-2)  # (B,...,T,n_out)
    base_t = base.placeholder
    out = tf.matmul(weights_t, base_t)  # (B,?,n_out)
    out = tf.reshape(out, shape[:rem_dims_start_axis] + rem_dims + [self.output.dim])  # (batch, ..., n_out)
    self.output.placeholder = out
    self.output.size_placeholder = {}

  def get_dep_layers(self):
    return super(GenericAttentionLayer, self).get_dep_layers() + [self.weights]

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    d.setdefault("from", [])
    super(GenericAttentionLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["weights"] = get_layer(d["weights"])

  @classmethod
  def get_out_data_from_opts(cls, base, weights, auto_squeeze=True, **kwargs):
    """
    :param LayerBase base:
    :param LayerBase weights:
    :param bool auto_squeeze:
    :rtype: Data
    """
    out = super(GenericAttentionLayer, cls).get_out_data_from_opts(base=base, **kwargs)
    base_rem_axes = base.output.get_axes(exclude_batch=True, exclude_time=True)
    base_rem_axes.remove(base.output.feature_dim_axis)
    weights_rem_axes = weights.output.get_axes(exclude_batch=True, exclude_time=True)
    # All the first axes in weights should match with base_rem_axes, just like the batch-dim.
    weights_rem_axes = weights_rem_axes[len(base_rem_axes):]
    if auto_squeeze:
      weights_rem_axes = [a for a in weights_rem_axes if weights.output.batch_shape[a] != 1]
    out.shape = (
      out.shape[:min(len(base_rem_axes), len(out.shape) - 1)] +
      tuple([weights.output.batch_shape[a] for a in weights_rem_axes]) + out.shape[-1:])
    return out


class DotAttentionLayer(GlobalAttentionContextBaseLayer):
  """
  Classic global attention: Dot-product as similarity measure between base_ctx and source.
  """

  layer_class = "dot_attention"

  def __init__(self, energy_factor=None, **kwargs):
    """
    :param LayerBase base: encoder output to attend on. defines output-dim
    :param LayerBase base_ctx: encoder output used to calculate the attention weights, combined with input-data.
      dim must be equal to input-data
    :param float|None energy_factor: the energy will be scaled by this factor.
      This is like a temperature for the softmax.
      In Attention-is-all-you-need, this is set to 1/sqrt(base_ctx.dim).
    """
    super(DotAttentionLayer, self).__init__(**kwargs)
    # We expect input_data of shape (batch, inner),
    # base_ctx of shape (batch, base_time, inner) and base of shape (batch, base_time, n_out).
    assert self.input_data.batch_ndim == 2
    assert self.input_data.time_dim_axis is None
    assert self.base.output.batch_ndim == 3
    assert self.base.output.dim == self.output.dim
    assert self.base_ctx.output.batch_ndim == 3
    assert self.input_data.dim == self.base_ctx.output.dim
    # And we want to do a dot product so that we get (batch, base_time).
    with tf.name_scope("att_energy"):
      # Get base of shape (batch, base_time, inner).
      base = self.base.output.get_placeholder_as_batch_major()  # (batch, base_time, n_out)
      base_seq_lens = self.base.output.get_sequence_lengths()
      base_ctx = self.base_ctx.output.get_placeholder_as_batch_major()  # (batch, base_time, inner)
      # Get source of shape (batch, inner, 1).
      source = tf.expand_dims(self.input_data.placeholder, axis=2)  # (batch, inner, 1)
      energy = tf.matmul(base_ctx, source)  # (batch, base_time, 1)
      energy.set_shape(tf.TensorShape([None, None, 1]))
      energy = tf.squeeze(energy, axis=2)  # (batch, base_time)
      if energy_factor:
        energy *= energy_factor
      # We must mask all values behind base_seq_lens. Set them to -inf, because we use softmax afterwards.
      energy_mask = tf.sequence_mask(base_seq_lens, maxlen=tf.shape(energy)[1])
      energy = tf.where(energy_mask, energy, float("-inf") * tf.ones_like(energy))
      self.base_weights = tf.nn.softmax(energy)  # (batch, base_time)
      base_weights_bc = tf.expand_dims(self.base_weights, axis=1)  # (batch, 1, base_time)
      out = tf.matmul(base_weights_bc, base)  # (batch, 1, n_out)
      out.set_shape(tf.TensorShape([None, 1, self.output.dim]))
      out = tf.squeeze(out, axis=1)  # (batch, n_out)
      self.output.placeholder = out
      self.output.size_placeholder = {}


class ConcatAttentionLayer(GlobalAttentionContextBaseLayer):
  """
  Additive attention / tanh-concat attention as similarity measure between base_ctx and source.
  This is used by Montreal, where as Stanford compared this to the dot-attention.
  The concat-attention is maybe more standard for machine translation at the moment.
  """

  layer_class = "concat_attention"

  def __init__(self, **kwargs):
    super(ConcatAttentionLayer, self).__init__(**kwargs)
    # We expect input_data of shape (batch, inner),
    # base_ctx of shape (batch, base_time, inner) and base of shape (batch, base_time, n_out).
    assert self.input_data.batch_ndim == 2
    assert self.input_data.time_dim_axis is None
    assert self.base.output.batch_ndim == 3
    assert self.base.output.dim == self.output.dim
    assert self.base_ctx.output.batch_ndim == 3
    assert self.input_data.dim == self.base_ctx.output.dim
    # And we want to get (batch, base_time).
    from TFUtil import expand_multiple_dims
    with tf.name_scope("att_energy"):
      # Get base of shape (batch, base_time, inner).
      base = self.base.output.get_placeholder_as_batch_major()  # (batch, base_time, n_out)
      base_seq_lens = self.base.output.get_sequence_lengths()
      base_ctx = self.base_ctx.output.get_placeholder_as_batch_major()  # (batch, base_time, inner)
      # Get source of shape (batch, inner, 1).
      source = tf.expand_dims(self.input_data.placeholder, axis=1)  # (batch, 1, inner)
      energy_in = tf.tanh(base_ctx + source)  # (batch, base_time, inner)
      energy_weights = self.add_param(tf.get_variable("v", shape=(self.input_data.dim,)))  # (inner,)
      energy_weights_bc = expand_multiple_dims(energy_weights, axes=(0, 1))  # (1, 1, inner)
      energy = tf.reduce_sum(energy_in * energy_weights_bc, axis=2)  # (batch, base_time)
      energy.set_shape(tf.TensorShape([None, None]))
      # We must mask all values behind base_seq_lens. Set them to -inf, because we use softmax afterwards.
      energy_mask = tf.sequence_mask(base_seq_lens, maxlen=tf.shape(energy)[1])
      energy = tf.where(energy_mask, energy, float("-inf") * tf.ones_like(energy))
      self.base_weights = tf.nn.softmax(energy)  # (batch, base_time)
      base_weights_bc = tf.expand_dims(self.base_weights, axis=1)  # (batch, 1, base_time)
      out = tf.matmul(base_weights_bc, base)  # (batch, 1, n_out)
      out.set_shape(tf.TensorShape([None, 1, self.output.dim]))
      out = tf.squeeze(out, axis=1)  # (batch, n_out)
      self.output.placeholder = out
      self.output.size_placeholder = {}


class GenericWindowAttentionLayer(AttentionBaseLayer):
  """
  """
  layer_class = "generic_window_attention"

  def __init__(self, weights, window_size, **kwargs):
    super(GenericWindowAttentionLayer, self).__init__(**kwargs)
    with tf.name_scope("base"):
      base = self.base.output.get_placeholder_as_time_major()

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
    from TFUtil import expand_dims_unbroadcast, dimshuffle

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
      # cidxs = tf.Print(cidxs, ["i=", self.network.layers[":i"].output.placeholder, "t=", t, "cidxs=", cidxs[window_size // 2:]])
      # We don't have multi_batch_beam for TF yet.
      # But tf.gather_nd or so might anyway be better to use here.
      # If that will not result in a sparse gradient in the while-loop,
      # some slicing with min(idxs)..max(idxs) might be anther option to at least reduce it a bit.
      # Note that gather_nd is broken up to TF 1.0 for this use case (see test_TFUtil.py),
      # so you need TF >=1.1 here.
      from TFUtil import assert_min_tf_version
      assert_min_tf_version((1, 1), "tf.gather_nd")
      batches_idxs = tf.range(tf.shape(cidxs)[1], dtype=tf.int32, name="batches_idxs")  # (batch,)
      batches_idxs_bc = expand_dims_unbroadcast(batches_idxs, axis=0, dim=tf.shape(cidxs)[0],
                                                name="batches_idxs_bc")  # (beam,batch)
      idxs_exp = tf.stack([cidxs, batches_idxs_bc], axis=2,
                          name="idxs_exp")  # (beam,batch,2), where the 2 stands for (base_time,batch)
      # Thus K == 2. gather_nd out will be idxs_exp.shape[:2] + params.shape[2:] = (beam,batch,n_in).
      gathered = tf.gather_nd(base, idxs_exp)  # (beam,batch,n_in)

    with tf.name_scope("gauss_window"):
      # Gauss window
      idxs_tr_bc = dimshuffle(idxs, (1, 0, 'x'))  # (batch,beam,inner_size)
      std_t_bc = dimshuffle(std, (0, 'x', 'x'))  # (batch,beam,inner_size)
      t_bc = dimshuffle(t, (0, 'x', 'x'))  # (batch,beam,inner_size)
      if inner_size:
        assert isinstance(inner_size, int)
        t_offs = tf.convert_to_tensor(
          [(i * inner_size_step - inner_size / 2.0) for i in range(inner_size)])  # (inner_size,)
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


class SelfAttentionLayer(_ConcatInputLayer):
  """
  Applies self-attention on the input. I.e., with input `x`,
  it will basically calculate

      att(Q x, K x, V x),

  where `att` is multi-head dot-attention for now, `Q`, `K`, `V` are matrices.
  """
  layer_class = "self_attention"
  recurrent = True

  def __init__(self, num_heads, total_key_dim, forward_weights_init="glorot_uniform", attention_dropout=0.0, **kwargs):
    """
    :param int num_heads:
    :param int total_key_dim:
    :param str forward_weights_init: see :func:`TFUtil.get_initializer`
    :param float attention_dropout:
    """
    super(SelfAttentionLayer, self).__init__(**kwargs)
    total_value_dim = self.output.dim
    assert total_key_dim % num_heads == 0, "must be divisible"
    assert total_value_dim % num_heads == 0, "must be divisible. total_value_dim = n_out"
    from TFUtil import get_initializer, dot, get_shape, move_axis
    with self.var_creation_scope():
      fwd_weights_initializer = get_initializer(
        forward_weights_init, seed=self.network.random.randint(2 ** 31), eval_local_ns={"layer": self})
      n_in = self.input_data.dim
      mat_n_out = total_key_dim * 2 + total_value_dim  # Q, K, V
      mat = self.add_param(tf.get_variable(
        name="QKV", shape=(n_in, mat_n_out), dtype=tf.float32, initializer=fwd_weights_initializer))
    x = self.input_data.placeholder
    if self.input_data.sparse:
      if x.dtype in [tf.uint8, tf.int8, tf.uint16, tf.int16]:
        x = tf.cast(x, tf.int32)
      x = tf.nn.embedding_lookup(mat, x)
    else:
      x = dot(x, mat)
    x.set_shape(tf.TensorShape(self.input_data.batch_shape_dense[:-1] + (mat_n_out,)))
    x_shape = [-1, -1, num_heads, mat_n_out // num_heads]
    assert self.input_data.batch_dim_axis in (0, 1)
    x_shape[self.input_data.batch_dim_axis] = tf.shape(x)[self.input_data.batch_dim_axis]
    x = tf.reshape(x, x_shape)  # (batch,time)|(time,batch) + (heads,qkv-dim//heads)
    x.set_shape(tf.TensorShape([None, None, num_heads, mat_n_out // num_heads]))
    assert self.input_data.batch_dim_axis in (0, 1)
    # (batch,heads,time,qkv-dim//heads)
    x = tf.transpose(x, [self.input_data.batch_dim_axis, 2, 1 - self.input_data.batch_dim_axis, 3])
    x.set_shape(tf.TensorShape([None, num_heads, None, mat_n_out // num_heads]))
    q, k, v = tf.split(
      x, [total_key_dim // num_heads, total_key_dim // num_heads, total_value_dim // num_heads], axis=-1, name="qkv")
    q *= (total_key_dim // num_heads) ** -0.5
    energy = tf.matmul(q, k, transpose_b=True)  # (batch,heads,time,time)
    energy_mask = tf.sequence_mask(
      self.input_data.get_sequence_lengths(), maxlen=tf.shape(energy)[-1])  # (batch,time)
    energy_mask = tf.reshape(energy_mask, [tf.shape(energy)[0], 1, 1, tf.shape(energy)[-1]])  # (batch,1,1,time)
    # Currently tf.where does not support broadcasting...
    energy_mask = tf.logical_and(energy_mask, tf.ones_like(energy, dtype=tf.bool))
    energy = tf.where(energy_mask, energy, float("-inf") * tf.ones_like(energy), name="energy_masked")
    weights = tf.nn.softmax(energy)  # (batch,heads,time,time)
    if attention_dropout:
      import TFUtil
      fn_train = lambda: TFUtil.dropout(
        weights,
        keep_prob=1 - attention_dropout,
        seed=self.network.random.randint(2 ** 31))
      fn_eval = lambda: weights
      weights = self.network.cond_on_train(fn_train, fn_eval)
    v = tf.matmul(weights, v)  # (batch,heads,time,v-dim//heads)
    v.set_shape(tf.TensorShape([None, num_heads, None, total_value_dim // num_heads]))
    v = tf.transpose(v, [0, 2, 1, 3])  # (batch,time,heads,v-dim//heads)
    if len(self.output.shape) == 2:
      v = tf.reshape(v, get_shape(v)[:2] + [total_value_dim])
      v.set_shape(tf.TensorShape([None, None, total_value_dim]))
    else:
      assert len(self.output.shape) == 1
      v = tf.reshape(v, get_shape(v)[:1] + [total_value_dim])  # (batch,v-dim)
      v.set_shape(tf.TensorShape([None, total_value_dim]))
    self.output.placeholder = v
    self.output.size_placeholder = self.input_data.size_placeholder.copy()

  @classmethod
  def get_out_data_from_opts(cls, n_out, name, sources, **kwargs):
    """
    :param int n_out:
    :param str name:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    assert sources
    import numpy
    out = sources[0].output.copy_as_batch_major().copy(name="%s_output" % name)
    if out.sparse:
      out.dtype = "float32"
      out.sparse = False
      out.shape = out.shape + (out.dim,)
    out.dim = n_out
    if len(out.shape) >= 2:
      if all(out.shape[:-1]):
        out.shape = (numpy.prod(out.shape[:-1]), n_out)
      else:
        out.shape = (None, n_out)
    else:
      out.shape = (n_out,)
    return out


class KenLmStateLayer(_ConcatInputLayer):
  """
  Get next word (or subword) each frame,
  accumulates string,
  keeps state of seen string so far,
  returns score (+log space, natural base e) of sequence,
  using KenLM (http://kheafield.com/code/kenlm/) (see :mod:`TFKenLM`).
  EOS (</s>) token must be used explicitly.
  """
  layer_class = "kenlm"
  recurrent = True

  def __init__(self, lm_file, vocab_file=None, vocab_unknown_label="UNK", bpe_merge_symbol=None,
               input_step_offset=0, dense_output=False,
               debug=False,
               **kwargs):
    """
    :param str|()->str lm_file: ARPA file or so. whatever KenLM supports
    :param str|None vocab_file: if the inputs are symbols, this must be provided. see :class:`Vocabulary`
    :param str vocab_unknown_label: for the vocabulary
    :param str|None bpe_merge_symbol: e.g. "@@" if you want to apply BPE merging
    :param int input_step_offset: if provided, will consider the input only from this step onwards
    :param bool dense_output: whether we output the score for all possible succeeding tokens
    :param bool debug: prints debug info
    """
    if callable(lm_file):
      lm_file = lm_file()
    import TFKenLM
    from TFUtil import expand_multiple_dims
    super(KenLmStateLayer, self).__init__(**kwargs)
    # Note: We later could extend it and have the state-behavior just as the :class:`CumsumLayer`.
    assert self._rec_previous_layer and self.input_data.time_dim_axis is None, (
      "%s: currently expected to run inside rec layer" % self)
    # Create KenLM handle. Use var scope to explicitly have it outside the loop.
    with self.var_creation_scope():
      self.lm_handle = TFKenLM.ken_lm_load(filename=lm_file)
    prev_step = self._rec_previous_layer.rec_vars_outputs["step"]
    next_step = prev_step + 1
    self.rec_vars_outputs["step"] = next_step
    new_input = self.input_data.placeholder
    input_dtype = tf.as_dtype(self.input_data.dtype)
    assert isinstance(input_dtype, tf.DType)
    self.vocab = None
    self.tf_vocab = None
    if vocab_file:
      with self.var_creation_scope():
        from GeneratingDataset import Vocabulary
        from TFNetwork import set_custom_post_init
        self.vocab = Vocabulary(vocab_file=vocab_file, unknown_label=vocab_unknown_label)
        assert self.input_data.sparse and self.vocab.num_labels == self.input_data.dim
        self.tf_vocab = tf.get_variable(
          name="vocab", shape=(self.vocab.num_labels,), dtype=tf.string, trainable=False,
          initializer=tf.zeros_initializer(tf.string))
        self.add_param(self.tf_vocab, saveable=False, trainable=False)
        set_custom_post_init(var=self.tf_vocab, func=self.vocab.tf_get_init_variable_func(var=self.tf_vocab))
    if input_dtype.is_integer:  # assume word-id in vocab
      assert self.tf_vocab, "%s: provide vocab_file" % self
      new_input = tf.gather(self.tf_vocab, indices=new_input) + " "
    else:
      assert input_dtype == tf.string
    assert new_input.dtype == tf.string
    if input_step_offset:
      new_input = tf.where(
        tf.greater_equal(prev_step, input_step_offset),
        new_input, tf.zeros_like(new_input))
    # See :class:`CumsumLayer` for comparison.
    prev_strings = self._rec_previous_layer.rec_vars_outputs["state"]
    next_strings = prev_strings + new_input
    self.rec_vars_outputs["state"] = next_strings
    prev_scores = self._rec_previous_layer.rec_vars_outputs["scores"]
    if dense_output:
      assert self.tf_vocab, "%s: provide vocab_file" % self
      new_abs_scores, new_abs_scores_dense = TFKenLM.ken_lm_abs_score_bpe_strings_dense(
        handle=self.lm_handle,
        bpe_merge_symbol=bpe_merge_symbol or "",
        strings=next_strings,
        labels=self.tf_vocab)
      new_abs_scores_bc = expand_multiple_dims(
        new_abs_scores, [i + new_abs_scores.get_shape().ndims for i in range(self.tf_vocab.get_shape().ndims)])
      new_rel_scores = new_abs_scores_dense - new_abs_scores_bc
    else:
      new_abs_scores = TFKenLM.ken_lm_abs_score_bpe_strings(
        handle=self.lm_handle,
        bpe_merge_symbol=bpe_merge_symbol or "",
        strings=next_strings)
      new_rel_scores = new_abs_scores - prev_scores
    if debug:
      # Print some info. Only for the first 3 steps because it will spam a lot.
      new_rel_scores = tf.cond(tf.less_equal(prev_step, 2), lambda: tf.Print(new_rel_scores, [
        str(self), "; step: ", prev_step,
        "; input shape: ", tf.shape(self.input_data.placeholder), str(self.input_data),
        "; input: ", self.input_data.placeholder,
        "; strings shape: ", tf.shape(next_strings),
        "; strings: ", "'" + next_strings + "'", "; new_abs_scores: ", new_abs_scores,
        "; sparse rel scores: ", new_abs_scores - prev_scores,
        "; min/max/mean rel scores: ", tf.reduce_min(new_rel_scores), "/", tf.reduce_max(new_rel_scores), "/", tf.reduce_mean(new_rel_scores)] +
        ["; vocab: ", self.tf_vocab] if self.tf_vocab else []),
        lambda: new_rel_scores)
    self.rec_vars_outputs["scores"] = new_abs_scores
    self.output.placeholder = new_rel_scores

  @classmethod
  def get_out_data_from_opts(cls, name, sources,
                             vocab_file=None, vocab_unknown_label="UNK", dense_output=False,
                             **kwargs):
    data = get_concat_sources_data_template(sources)
    dtype = tf.as_dtype(data.dtype)
    assert isinstance(dtype, tf.DType)
    assert (data.sparse and dtype.is_integer) or dtype == tf.string
    data = data.copy(name="%s_output" % name)
    data.dtype = "float32"
    data.sparse = False
    if dense_output:
      from GeneratingDataset import Vocabulary
      vocab = Vocabulary(vocab_file=vocab_file, unknown_label=vocab_unknown_label)
      data.dim = vocab.num_labels
      data.shape = data.shape + (vocab.num_labels,)
    else:
      data.dim = None
    return data

  @classmethod
  def get_rec_initial_extra_outputs(cls, batch_dim, rec_layer, sources=(), **kwargs):
    data = get_concat_sources_data_template(sources)
    # Assume inside RecLayer.
    assert all(data.shape)
    batch_shape = data.get_batch_shape(batch_dim=batch_dim)
    return {
      "state": tf.zeros(batch_shape, dtype=tf.string),
      "step": tf.constant(0, dtype=tf.int32),
      "scores": tf.zeros(batch_shape, dtype=tf.float32)}


class BaseRNNCell(rnn_cell.RNNCell):
  """
  Extends :class:`rnn_cell.RNNCell` by having explicit static attributes describing some properties.
  """

  def get_input_transformed(self, x):
    """
    Usually the cell itself does the transformation on the input.
    However, it would be faster to do it outside the recurrent loop.
    This function will get called outside the loop.

    :param tf.Tensor x: (time, batch, dim)
    :return: like x, maybe other feature-dim
    :rtype: tf.Tensor
    """
    return x


class RHNCell(BaseRNNCell):
  """
  Recurrent Highway Layer.
  With optional dropout for recurrent state (fixed over all frames - some call this variational).

  References:
    https://github.com/julian121266/RecurrentHighwayNetworks/
    https://arxiv.org/abs/1607.03474
  """

  def __init__(self, num_units, is_training=None, depth=5, dropout=0.0, dropout_seed=None, transform_bias=None,
               batch_size=None):
    """
    :param int num_units:
    :param bool|tf.Tensor|None is_training:
    :param int depth:
    :param float dropout:
    :param int dropout_seed:
    :param float|None transform_bias:
    :param int|tf.Tensor|None batch_size:
    """
    from TFNetwork import TFNetwork
    super(RHNCell, self).__init__()
    self._num_units = num_units
    if is_training is None:
      is_training = TFNetwork.get_current_network().train_flag
    self.is_training = is_training
    self.depth = depth
    self.dropout = dropout
    if dropout_seed is None:
      dropout_seed = TFNetwork.get_current_network().random.randint(2 ** 31)
    self.dropout_seed = dropout_seed
    self.transform_bias = transform_bias or 0.0
    self.batch_size = batch_size
    self._dropout_mask = None

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  @staticmethod
  def _linear(x, output_dim):
    """
    :param tf.Tensor x:
    :param int output_dim:
    :rtype: tf.Tensor
    """
    from TFUtil import var_creation_scope, dot
    input_dim = x.get_shape().dims[-1].value
    assert input_dim is not None, "%r shape unknown" % (x,)
    with var_creation_scope():
      weights = tf.get_variable("W", shape=(input_dim, output_dim))
    x = dot(x, weights)
    return x

  def _get_dropout_mask(self):
    if self._dropout_mask is not None:
      return self._dropout_mask

    from TFUtil import var_creation_scope, cond
    # Create the dropout masks outside the loop:
    with var_creation_scope():
      def get_mask():
        if self.batch_size is not None:
          batch_size = self.batch_size
        else:
          from TFNetworkLayer import LayerBase
          batch_size = LayerBase.get_recent_layer().get_batch_dim()
        keep_prob = 1.0 - self.dropout
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += tf.random_uniform((batch_size, self._num_units), seed=self.dropout_seed, dtype=tf.float32)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = tf.floor(random_tensor)
        return binary_tensor * (1.0 / keep_prob)
      self._dropout_mask = cond(self.is_training, get_mask, lambda: 1.0)
    return self._dropout_mask

  def _optional_dropout(self, state):
    if not self.dropout:
      return state
    if self.is_training is False:
      return state
    state *= self._get_dropout_mask()
    state.set_shape((None, self._num_units))
    return state

  def get_input_transformed(self, x):
    """
    :param tf.Tensor x: (time, batch, dim)
    :return: (time, batch, num_units * 2)
    :rtype: tf.Tensor
    """
    from TFUtil import var_creation_scope
    x = self._linear(x, self._num_units * 2)
    with var_creation_scope():
      bias = tf.get_variable(
        "b", shape=(self._num_units * 2,),
        initializer=tf.constant_initializer(
          [0.0] * self._num_units + [self.transform_bias] * self._num_units))
    x += bias
    return x

  def call(self, inputs, state):
    """
    :param tf.Tensor inputs:
    :param tf.Tensor state:
    :return: (output, state)
    :rtype: (tf.Tensor, tf.Tensor)
    """
    inputs.set_shape((None, self._num_units * 2))
    state.set_shape((None, self._num_units))

    # Carry-gate coupled with transform gate: C = 1 - T
    current_state = state
    for i in range(self.depth):
      current_state_masked = self._optional_dropout(current_state)
      with tf.variable_scope('depth_%i' % i):
        state_transformed = self._linear(current_state_masked, self._num_units * 2)
      if i == 0:
        state_transformed += inputs
      h, t = tf.split(state_transformed, 2, axis=-1)
      h = tf.tanh(h)
      t = tf.sigmoid(t)
      # Simplified equation for better numerical stability.
      # The current_state here should be without the dropout applied,
      # because dropout would divide by keep_prop, which can blow up the state.
      current_state += t * (h - current_state)

    return current_state, current_state
