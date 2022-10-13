
"""
Defines multiple recurrent layers, most importantly :class:`RecLayer`.
"""

from __future__ import print_function

import typing
import tensorflow as tf
import returnn.tf.compat as tf_compat
try:
  from tensorflow.python.ops.nn import rnn_cell
except ImportError:
  from tensorflow.python.ops import rnn_cell
from returnn.tf.network import LayerNotFound, TFNetwork
from .basic import LayerBase, InternalLayer, _ConcatInputLayer, SearchChoices, get_concat_sources_data_template, Loss
from returnn.tf.util.data import Data, Dim, SpatialDim, FeatureDim, SearchBeam
from returnn.tf.util.basic import reuse_name_scope
from returnn.tf.util import basic as tf_util
from returnn.util.basic import NotSpecified
from returnn.log import log


class RecLayer(_ConcatInputLayer):
  """
  Recurrent layer, has support for several implementations of LSTMs (via ``unit`` argument),
  see :ref:`tf_lstm_benchmark` (https://returnn.readthedocs.io/en/latest/tf_lstm_benchmark.html),
  and also GRU, or simple RNN.
  Via `unit` parameter, you specify the operation/model performed in the recurrence.
  It can be a string and specify a RNN cell, where all TF cells can be used,
  and the `"Cell"` suffix can be omitted; and case is ignored.
  Some possible LSTM implementations are (in all cases for both CPU and GPU):

   * BasicLSTM (the cell), via official TF, pure TF implementation
   * LSTMBlock (the cell), via tf.contrib.rnn.
   * LSTMBlockFused, via tf.contrib.rnn. should be much faster than BasicLSTM
   * CudnnLSTM, via tf.contrib.cudnn_rnn. This is experimental yet.
   * NativeLSTM, our own native LSTM. should be faster than LSTMBlockFused.
   * NativeLstm2, improved own native LSTM, should be the fastest and most powerful.

  We default to the current tested fastest one, i.e. NativeLSTM.
  Note that they are currently not compatible to each other, i.e. the way the parameters are represented.

  A subnetwork can also be given which will be evaluated step-by-step,
  which can use attention over some separate input,
  which can be used to implement a decoder in a sequence-to-sequence scenario.
  The subnetwork will get the extern data from the parent net as templates,
  and if there is input to the RecLayer,
  then it will be available as the "source" data key in the subnetwork.
  The subnetwork is specified as a `dict` for the `unit` parameter.
  In the subnetwork, you can access outputs from layers from the previous time step when they
  are referred to with the "prev:" prefix.

  Example::

      {
          "class": "rec",
          "from": "input",
          "unit": {
            # Recurrent subnet here, operate on a single time-step:
            "output": {
              "class": "linear",
              "from": ["prev:output", "data:source"],
              "activation": "relu",
              "n_out": n_out},
          },
          "n_out": n_out},
      }

  More examples can be seen in :mod:`test_TFNetworkRecLayer` and :mod:`test_TFEngine`.

  The subnetwork can automatically optimize the inner recurrent loop
  by moving layers out of the loop if possible.
  It will try to do that greedily. This can be disabled via the option `optimize_move_layers_out`.
  It assumes that those layers behave the same with time-dimension or without time-dimension and used per-step.
  Examples for such layers are :class:`LinearLayer`, :class:`RnnCellLayer`
  or :class:`SelfAttentionLayer` with option `attention_left_only`.

  This layer can also be inside another RecLayer. In that case, it behaves similar to :class:`RnnCellLayer`.
  (This support is somewhat incomplete yet. It should work for the native units such as NativeLstm.)

  Also see :ref:`recurrency`.
  """

  layer_class = "rec"
  recurrent = True
  _default_lstm_unit = "NativeLSTM2"  # TFNativeOp.NativeLstmCell
  SubnetworkRecCell = None  # type: typing.Type[_SubnetworkRecCell]  # set below

  def __init__(self,
               unit="lstm", unit_opts=None,
               direction=None, input_projection=True,
               initial_state=None,
               max_seq_len=None, max_seq_len_via=None,
               forward_weights_init=None, recurrent_weights_init=None, bias_init=None,
               optimize_move_layers_out=None,
               cheating=False,
               unroll=False, back_prop=None,
               use_global_rec_step_offset=False,
               include_eos=False,
               debug=None,
               axis=None,
               **kwargs):
    """
    :param str|_SubnetworkRecCell unit: the RNNCell/etc name, e.g. "nativelstm". see comment below.
      alternatively a whole subnetwork, which will be executed step by step,
      and which can include "prev" in addition to "from" to refer to previous steps.
      The subnetwork is specified as a net dict in the config.
    :param None|dict[str] unit_opts: passed to RNNCell creation
    :param int|None direction: None|1 -> forward, -1 -> backward
    :param bool input_projection: True -> input is multiplied with matrix. False only works if same input dim
    :param LayerBase|str|float|int|tuple|None initial_state:
    :param int|tf.Tensor|None max_seq_len: if unit is a subnetwork. str will be evaluated. see code
    :param LayerBase|None max_seq_len_via: like max_seq_len but via another layer
    :param str forward_weights_init: see :func:`returnn.tf.util.basic.get_initializer`
    :param str recurrent_weights_init: see :func:`returnn.tf.util.basic.get_initializer`
    :param str bias_init: see :func:`returnn.tf.util.basic.get_initializer`
    :param bool|None optimize_move_layers_out: will automatically move layers out of the loop when possible
    :param bool cheating: Unused, is now part of ChoiceLayer
    :param bool unroll: if possible, unroll the loop (implementation detail)
    :param bool|None back_prop: for tf.while_loop. the default will use self.network.train_flag
    :param bool use_global_rec_step_offset:
    :param bool include_eos: for search, whether we should include the frame where "end" is True
    :param bool|None debug:
    :param Dim|str axis: specify the axis to iterate over.
      It can also be the special marker ``single_step_dim``, or an outer recurrent time dim.
    """
    super(RecLayer, self).__init__(**kwargs)
    import re
    from returnn.tf.util.basic import is_gpu_available_in_session
    rnn_contrib = None
    try:
      # noinspection PyUnresolvedReferences
      from tensorflow.contrib import rnn as rnn_contrib
    except ImportError:
      pass
    from tensorflow.python.util import nest
    cudnn_rnn = None
    if is_gpu_available_in_session():
      try:
        # noinspection PyUnresolvedReferences
        from tensorflow.contrib import cudnn_rnn
      except ImportError:
        pass
    import returnn.tf.native_op as tf_native_op
    if direction is not None:
      assert direction in [-1, 1]
    self._last_hidden_state = None  # type: typing.Optional[tf.Tensor]
    self._direction = direction
    self._initial_state_deps = [layer for layer in nest.flatten(initial_state) if isinstance(layer, LayerBase)]
    self._input_projection = input_projection
    self._max_seq_len_via = max_seq_len_via
    if max_seq_len_via:
      assert max_seq_len is None
      assert max_seq_len_via.output.batch_shape == (), "%s: max_seq_len_via %s not scalar" % (self, max_seq_len_via)
      max_seq_len = max_seq_len_via.output.placeholder
    self._max_seq_len = max_seq_len
    # Note: Most logic and preprocessing on axis happens already in transform_config_dict.
    # Specifically, it should always be set and is always a Dim instance here.
    # It can be single_step_dim.
    from returnn.tf.util.data import single_step_dim
    assert isinstance(axis, Dim)
    if axis == single_step_dim:
      axis = None  # this indicates in the following that we do a single-step operation
    if axis and self.input_data and self.input_data.have_dim_tag(axis):
      axis_int = self.input_data.get_axis_from_description(axis)
      self.input_data.time_dim_axis = axis_int  # makes some of the following code easier
      if self.input_data.time_dim_axis == self.input_data.feature_dim_axis:
        self.input_data.feature_dim_axis = NotSpecified
    self.time_dim_tag = axis
    self.include_eos = include_eos
    if optimize_move_layers_out is None:
      optimize_move_layers_out = self.network.get_config().bool("optimize_move_layers_out", True)
    self._optimize_move_layers_out = optimize_move_layers_out
    if cheating:
      print("Warning: cheating is an unused parameter in RecLayer, "
            "to enable cheating set the flag in a ChoiceLayer instead.", file=log.v2)
    self._unroll = unroll
    if back_prop is None:
      back_prop = self.network.train_flag is not False
    self.back_prop = back_prop
    self._use_global_rec_step_offset = use_global_rec_step_offset
    if debug is None:
      debug = self.network.get_config().bool("debug_rec_layer", False)
    self.debug = debug
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
    #   https://www.tensorflow.org/api_docs/python/tf/initializers
    #   https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Orthogonal
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
    from returnn.tf.util.basic import get_initializer, xavier_initializer
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
    scope = "rec" if self.name_scope is None else tf_compat.v1.get_variable_scope()
    with reuse_name_scope(scope, initializer=default_var_initializer) as scope:
      assert isinstance(scope, tf_compat.v1.VariableScope)
      self._rec_scope = scope
      scope_name_prefix = scope.name + "/"  # e.g. "layer1/rec/"
      with self.var_creation_scope():
        self._initial_state = None
        if self._rec_previous_layer:  # inside another RecLayer
          self._initial_state = self._rec_previous_layer.rec_vars_outputs["state"]
        elif initial_state is not None:
          if initial_state:
            assert isinstance(unit, str), 'initial_state not supported currently for custom unit'
          self._initial_state = RnnCellLayer.get_rec_initial_state(
            initial_state=initial_state, n_out=self.output.dim, unit=unit, unit_opts=unit_opts,
            batch_dim=self.network.get_data_batch_dim(), name=self.name,
            rec_layer=self)
        self.cell = self._get_cell(unit, unit_opts=unit_opts)
        base_types = (rnn_cell.RNNCell,)
        if rnn_contrib:
          # noinspection PyUnresolvedReferences
          base_types += (rnn_contrib.FusedRNNCell, rnn_contrib.LSTMBlockWrapper)
        cudnn_types = None
        if cudnn_rnn:
          # noinspection PyUnresolvedReferences
          cudnn_types = (cudnn_rnn.CudnnLSTM, cudnn_rnn.CudnnGRU)
        if isinstance(self.cell, base_types):
          y = self._get_output_cell(self.cell)
        elif cudnn_rnn and isinstance(self.cell, cudnn_types):
          y = self._get_output_cudnn(self.cell)
        elif isinstance(self.cell, tf_native_op.RecSeqCellOp):
          y = self._get_output_native_rec_op(self.cell)
        elif isinstance(self.cell, _SubnetworkRecCell):
          y = self._get_output_subnet_unit(self.cell)
        else:
          raise Exception("invalid type: %s" % type(self.cell))
        if self._rec_previous_layer:  # inside another RecLayer
          self.rec_vars_outputs["state"] = self._last_hidden_state
        self.output.placeholder = y
        # Very generic way to collect all created params.
        # Note that for the TF RNN cells, there is no other way to do this.
        # Also, see the usage of :func:`LayerBase.cls_layer_scope`, e.g. for initial vars.
        params = tf_compat.v1.get_collection(
          tf_compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=re.escape(scope_name_prefix))
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
      self.add_param(p)

      # Sublayers do not know whether the RecLayer is trainable. If it is not, we need to mark all defined parameters
      # as untrainable
      if not self.trainable:
        trainable_collection_ref = p.graph.get_collection_ref(tf_compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        if p in trainable_collection_ref:
          trainable_collection_ref.remove(p)

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    ls = super(RecLayer, self).get_dep_layers()
    ls += self._initial_state_deps
    if isinstance(self.cell, _SubnetworkRecCell):
      ls += self.cell.get_parent_deps()
      if self.cell.input_layers_net and "output" in self.cell.input_layers_net.layers:
        ls.append(self.cell.input_layers_net.layers["output"])
      elif self.cell.output_layers_net and "output" in self.cell.output_layers_net.layers:
        ls.append(self.cell.output_layers_net.layers["output"])
    if self._max_seq_len_via:
      ls.append(self._max_seq_len_via)
    return ls

  @classmethod
  def transform_source_and_axis(cls, network, source_data=None, have_dyn_seq_len_end=False, axis=None, opts=None):
    """
    :param returnn.tf.network.TFNetwork network:
    :param Data|None source_data:
    :param bool have_dyn_seq_len_end:
    :param Dim|str|None axis:
    :param dict[str]|None opts:
    :rtype: (Data|None, Dim)
    """
    from returnn.tf.util.data import single_step_dim
    name = opts["_name"] if opts and "_name" in opts else "unknown"
    if source_data and isinstance(axis, str):
      axis = source_data.get_dim_tag_from_description(axis)
    if not axis:
      if opts and opts.get("state"):
        axis = single_step_dim
      elif source_data and not source_data.have_time_axis():  # expect to be inside other RecLayer
        axis = single_step_dim
      else:
        # We will output a time-dim.
        if source_data and not have_dyn_seq_len_end:
          assert source_data.have_time_axis()
          axis = source_data.get_time_dim_tag()
        elif opts and opts.get("size_target"):  # if this is set, always use it
          target_data = cls._static_get_target_value(
            target=opts["size_target"], mark_data_key_as_used=True, network=network)
          assert target_data.have_time_axis()
          axis = target_data.get_time_dim_tag()
        elif opts and opts.get("target") and network.eval_flag and not network.search_flag:
          target_data = cls._static_get_target_value(
            target=opts["target"][0] if isinstance(opts["target"], (list, tuple)) else opts["target"],
            mark_data_key_as_used=True, network=network)
          assert target_data.have_time_axis()
          axis = target_data.get_time_dim_tag()
        else:
          # We create a new time-dim.
          # Usually the loop would be dynamic via an "end" layer which defines the end-of-sequence (EOS).
          # However, there are cases such as the RecUnstackLayer which can also define the time dim.
          # Expect that we have a subnet.
          assert have_dyn_seq_len_end or (opts and isinstance(opts.get("unit"), dict))
          axis = SpatialDim("dyn-time:%s%s" % (network.get_absolute_name_prefix(), name), auto_generated=True)
    assert isinstance(axis, Dim)
    inside_rec_time_dim = network.get_inside_rec_time_dim(inside_loop=True)
    over_rec_time_dim = network.get_inside_rec_time_dim(inside_loop=False)
    if axis == inside_rec_time_dim:
      axis = single_step_dim
    if axis == single_step_dim:
      if over_rec_time_dim and not inside_rec_time_dim:  # moved out of loop
        axis = over_rec_time_dim
    if source_data and source_data.have_dim_tag(axis):
      # Make sure it is marked as time dim. This will make it easier in the following.
      source_data.time_dim_axis = source_data.get_axis_from_description(axis)
      if source_data.time_dim_axis == source_data.feature_dim_axis:
        source_data.feature_dim_axis = NotSpecified
    return source_data, axis

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    This method transforms the templates in the config dictionary into references
    of the layer instances (and creates them in the process).

    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    if isinstance(d.get("unit"), dict):
      d["n_out"] = d.get("n_out", NotSpecified)  # disable automatic guessing
    super(RecLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)  # everything except "unit"
    if "initial_state" in d:
      d["initial_state"] = RnnCellLayer.transform_initial_state(
        d["initial_state"], network=network, get_layer=get_layer)

    # We need to figure out the output time dim tag at this early point,
    # because _SubnetworkRecCell might need it during template construction.
    source_data = get_concat_sources_data_template(d["sources"]) if d["sources"] else None
    time_dim_tag = d.get("axis")
    have_dyn_seq_len_end = False
    if isinstance(d.get("unit"), dict):
      have_dyn_seq_len_end = "end" in d["unit"]
    source_data, time_dim_tag = cls.transform_source_and_axis(
      network=network, source_data=source_data, have_dyn_seq_len_end=have_dyn_seq_len_end, opts=d,
      axis=time_dim_tag)
    d["axis"] = time_dim_tag

    if isinstance(d.get("unit"), dict):
      sub_net_dict = d.pop("unit")
      assert time_dim_tag  # expect to always have this (also not being inside other rec layer)
      subnet = cls.SubnetworkRecCell(
        parent_net=network, parent_get_layer=get_layer,
        source_data=source_data, time_dim_tag=time_dim_tag,
        net_dict=sub_net_dict, rec_layer_name=d["_name"])
      d["unit"] = subnet

    if d.get("max_seq_len_via"):
      d["max_seq_len_via"] = get_layer(d["max_seq_len_via"])

    if isinstance(d.get("max_seq_len"), str):

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

      # Note: Normally we do not expect that anything is added to the TF computation graph
      # within transform_config_dict, so this is kind of bad practice.
      # However, we must make sure at this point that any layers will get resolved via get_layer calls.
      # Also make sure that we do not introduce any new name-scope here
      # as this would confuse recursive get_layer calls.
      d["max_seq_len"] = eval(d["max_seq_len"], {"max_len_from": max_len_from, "tf": tf})

  @classmethod
  def get_out_data_from_opts(cls, name, network, sources, unit, axis=None, out_dim=None, initial_state=None,
                             **kwargs):
    """
    :param str name:
    :param returnn.tf.network.TFNetwork network:
    :param list[LayerBase] sources:
    :param str|dict[str] unit:
    :param Dim axis:
    :param Dim|None out_dim:
    :param str|LayerBase|list[str|LayerBase] initial_state:
    :rtype: Data
    """
    from tensorflow.python.util import nest
    source_data = get_concat_sources_data_template(sources) if sources else None
    # Note: We preprocessed axis in transform_config_dict, so it can only be a Dim instance here.
    from returnn.tf.util.data import single_step_dim
    assert isinstance(axis, Dim)
    if source_data and source_data.have_dim_tag(axis):
      # This will make it easier in the following.
      source_data.time_dim_axis = source_data.get_axis_from_description(axis)
      if source_data.time_dim_axis == source_data.feature_dim_axis:
        source_data.feature_dim_axis = NotSpecified
    n_out = kwargs.get("n_out", NotSpecified)
    out_type = kwargs.get("out_type", None)
    loss = kwargs.get("loss", None)
    deps = list(sources)  # type: typing.List[LayerBase]
    deps += [layer for layer in nest.flatten(initial_state) if isinstance(layer, LayerBase)]
    if isinstance(unit, _SubnetworkRecCell):  # subnetwork
      subnet = unit
      out = subnet.layer_data_templates["output"].output.copy_template(name="%s_output" % name)
      if axis != single_step_dim:
        out = out.copy_add_dim_by_tag(dim_tag=axis, axis=0, unbroadcast=True)
        out.time_dim_axis = 0
      out = out.copy_template_set_ctx(network.get_control_flow_ctx())
      deps += subnet.get_parent_deps()
    elif out_type or n_out is not NotSpecified or out_dim or loss:
      assert source_data
      out = source_data.copy_template(name="%s_output" % name)
      if out.sparse:
        out.dtype = "float32"
        out.sparse = False
        out = out.copy_add_feature_dim()  # dummy
      if not out_dim:
        if n_out is NotSpecified or not n_out:
          assert out_type and "dim" in out_type
          n_out = out_type["dim"]
        out_dim = FeatureDim("%s:feature" % name, dimension=n_out, auto_generated=True)
      if out.have_feature_axis():
        out = out.copy_template_replace_dim_tag(axis=out.feature_dim_axis, new_dim_tag=out_dim)
      else:
        out = out.copy_add_dim_by_tag(dim_tag=out_dim, unbroadcast=True, axis=-1)
        out.feature_dim_axis = NotSpecified
      if axis != single_step_dim and not out.have_dim_tag(axis):
        out = out.copy_add_dim_by_tag(axis, unbroadcast=True)
      if out.have_dim_tag(axis):
        out.time_dim_axis = out.get_axis_from_description(axis)
      if out.have_time_axis() and axis == out.get_time_dim_tag():
        out = out.copy_as_time_batch_major()
      elif out.have_batch_axis():
        # We expect to be inside another RecLayer, and should do a single step (like RnnCellLayer).
        out = out.copy_as_batch_major()  # The output is then [B,F]
    else:
      raise Exception("n_out or out_type or out_dim must be specified")
    if n_out is not NotSpecified:
      assert n_out == out.dim
    if out_dim:
      assert out_dim == out.feature_dim_or_sparse_dim
    if out_type:
      for k, v in out_type.items():
        assert getattr(out, k) == v
    for dep in deps:
      if dep:
        out.beam = SearchBeam.get_combined_beam(out.beam, dep.output.beam)
    return out

  def get_absolute_name_scope_prefix(self):
    """
    :rtype: str
    """
    if self.name_scope is not None:
      return self.get_base_absolute_name_scope_prefix()
    return self.get_base_absolute_name_scope_prefix() + "rec/"  # all under "rec" sub-name-scope

  @classmethod
  def get_rec_initial_extra_outputs(cls, **kwargs):
    """
    :rtype: dict[str,tf.Tensor|tuple[tf.Tensor]]
    """
    # axis is handled in transform_config_dict
    from returnn.tf.util.data import single_step_dim
    axis = kwargs["axis"]
    assert isinstance(axis, Dim)
    if axis != single_step_dim:
      return {}
    # We expect to be inside another RecLayer, and should do a single step (like RnnCellLayer).
    return {"state": RnnCellLayer.get_rec_initial_state(**kwargs)}

  @classmethod
  def get_rec_initial_output(cls, **kwargs):
    """
    :rtype: tf.Tensor
    """
    # This is only called if we are inside another rec layer.
    return RnnCellLayer.get_rec_initial_output(**kwargs)

  _rnn_cells_dict = {}

  @classmethod
  def _create_rnn_cells_dict(cls):
    import returnn.tf.native_op as tf_native_op
    from returnn.tf.util.basic import is_gpu_available_in_session
    allowed_types = (rnn_cell.RNNCell, tf_native_op.RecSeqCellOp)
    rnn_contrib = None
    try:
      # noinspection PyUnresolvedReferences
      from tensorflow.contrib import rnn as rnn_contrib
      allowed_types += (rnn_contrib.FusedRNNCell, rnn_contrib.LSTMBlockWrapper)
    except ImportError:
      pass
    cudnn_rnn = None
    if is_gpu_available_in_session():
      try:
        # noinspection PyUnresolvedReferences
        from tensorflow.contrib import cudnn_rnn
        allowed_types += (cudnn_rnn.CudnnLSTM, cudnn_rnn.CudnnGRU)
      except ImportError:
        pass

    # noinspection PyShadowingNames
    def maybe_add(key, v):
      """
      :param str key:
      :param type v:
      """
      if v is BaseRNNCell:
        return
      if isinstance(v, type) and issubclass(v, allowed_types):
        name = key
        if name.endswith("Cell"):
          name = name[:-len("Cell")]
        name = name.lower()
        assert cls._rnn_cells_dict.get(name) in [v, None]
        cls._rnn_cells_dict[name] = v

    for key, v in globals().items():
      maybe_add(key, v)
    for key, v in vars(rnn_cell).items():
      maybe_add(key, v)
    if rnn_contrib:
      for key, v in vars(rnn_contrib).items():
        maybe_add(key, v)
    for key, v in vars(tf_native_op).items():
      maybe_add(key, v)
    if cudnn_rnn:
      for key, v in vars(cudnn_rnn).items():
        maybe_add(key, v)
    # Alias for the standard LSTM cell, because self._get_cell(unit="lstm") will use "NativeLSTM" by default.
    maybe_add("StandardLSTM", rnn_cell.LSTMCell)

  _warn_msg_once_for_cell_name = set()

  @classmethod
  def get_rnn_cell_class(cls, name, cell_only=False):
    """
    :param str|type name: cell name, minus the "Cell" at the end
    :param bool cell_only: i.e. for single-step execution
    :rtype: type[rnn_cell.RNNCell]|type[returnn.tf.native_op.RecSeqCellOp]
    """
    if isinstance(name, type):
      from returnn.tf import native_op as tf_native_op
      assert issubclass(name, (rnn_cell.RNNCell, tf_native_op.RecSeqCellOp))
      return name
    if not cls._rnn_cells_dict:
      cls._create_rnn_cells_dict()
    # We have some automatic replacement logic here.
    # In our CustomCheckpointLoader, we have logic to automatically convert one param format into another.
    # Be careful though that the defaults still might lead to different computations.
    # E.g. StandardLSTM/BasicLSTM use forget_bias=1 by default, which is not used for param initialization,
    # but will always be added.
    # Thus when changing from e.g. NativeLSTM -> StandardLSTM, param importing works,
    # but you explicitly need to specify `"unit_opts": {"forget_bias": 0.0}`, otherwise it will be wrong.
    from returnn.tf.util.basic import is_gpu_available_in_session
    if not is_gpu_available_in_session():
      m = {"cudnnlstm": "LSTMBlockFused", "cudnngru": "GRUBlock"}
      if name.lower() in m:
        if name.lower() not in cls._warn_msg_once_for_cell_name:
          print("You have selected unit %r in a rec layer which is for GPU only, so we are using %r instead." %
                (name, m[name.lower()]), file=log.v2)
          cls._warn_msg_once_for_cell_name.add(name.lower())
        name = m[name.lower()]
    if name.lower() in ["lstmp", "lstm"]:
      name = cls._default_lstm_unit
    if not tf_compat.have_contrib and name.lower() in ["LSTMBlock".lower(), "LSTMBlockFused".lower()]:
      # E.g. TF 2 does not have the contrib module and also does not have LSTMBlock anymore.
      # (Actually, the raw op is still there, but not the wrapper class...)
      if cell_only:
        name = "StandardLSTM"
      else:
        name = "NativeLSTM2"
    if name.lower() == "BasicLSTM".lower() and name.lower() not in cls._rnn_cells_dict:
      # TF 2 does not have BasicLSTM anymore. Use StandardLSTM instead.
      name = "StandardLSTM"
    if name.lower() not in cls._rnn_cells_dict:
      raise Exception("unknown cell %r. known cells: %r" % (name, sorted(cls._rnn_cells_dict.keys())))
    return cls._rnn_cells_dict[name.lower()]

  def _get_input(self, strict=True):
    """
    :param bool strict:
    :return: (in_data, x, seq_len), where x is (time,batch,...,dim) and seq_len is (batch,)
    :rtype: (Data, tf.Tensor, tf.Tensor)
    """
    assert self.input_data
    in_data = self.input_data
    if not self.time_dim_tag:
      in_data = in_data.copy_add_spatial_dim(spatial_dim_axis=0)
      in_data.time_dim_axis = 0
    assert in_data.have_time_axis(), (
      "%s, input %s should have time dim now, time dim tag %s" % (self, in_data, self.time_dim_tag))
    if strict:
      # Merge all other axes except time and feature such that we get (time,batch',dim).
      in_data = in_data.copy_as_time_major()
      if not in_data.sparse:
        in_data = in_data.copy_with_feature_last()
    else:
      if in_data.have_batch_axis():
        in_data = in_data.copy_as_time_batch_major()
      else:
        in_data = in_data.copy_as_time_major()
    in_data_ = in_data
    if strict and in_data_.batch_ndim_dense != 3:
      assert in_data_.batch_ndim_dense > 3
      batch_axes = list(range(1, in_data_.batch_ndim_dense - 1))
      assert len(batch_axes) >= 2
      in_data_ = in_data_.copy_merge_into_batch(batch_axes)
    return in_data, in_data_.placeholder, in_data_.get_sequence_lengths()

  def _post_proc_output_cell_strict(self, y, in_data):
    """
    :param tf.Tensor y: (time,batch,dim)
    :param Data in_data:
    :rtype: tf.Tensor
    :return: (time,batch,dim) or (time,batch,...,dim)
    """
    if in_data.batch_ndim_dense > 3:
      y_shape = tf_util.get_shape(in_data.placeholder)
      if not in_data.sparse:
        y_shape = y_shape[:-1]
      y_shape += [tf_util.get_shape_dim(y, -1)]
      y = tf.reshape(y, y_shape)
    out_data = in_data.copy_template_dense()
    out_data = out_data.copy_template_replace_dim_tag(
      axis=out_data.feature_dim_axis, new_dim_tag=self.output.feature_dim_or_sparse_dim)
    out_data.placeholder = y
    if not self.time_dim_tag:
      out_data = out_data.copy_squeeze_axes(axes=[0])
    # The output format should match now.
    # If this is not the case, we should fix get_out_data_from_opts accordingly
    # and avoid unnecessary further transformations here, esp any transposes.
    assert out_data.dim_tags == self.output.dim_tags
    return out_data.placeholder

  @classmethod
  def get_losses(cls, name, network, output, loss=None, reduce_func=None, layer=None, **kwargs):
    """
    :param str name: layer name
    :param returnn.tf.network.TFNetwork network:
    :param Loss|None loss: argument just as for __init__
    :param Data output: the output (template) for the layer
    :param ((tf.Tensor)->tf.Tensor)|None reduce_func:
    :param LayerBase|None layer:
    :param kwargs: other layer kwargs
    :rtype: list[returnn.tf.network.LossHolder]
    """
    from returnn.tf.network import LossHolder
    losses = super(RecLayer, cls).get_losses(
      name=name, network=network, output=output, loss=loss, layer=layer, reduce_func=reduce_func, **kwargs)
    unit = kwargs["unit"]
    if isinstance(unit, _SubnetworkRecCell):  # subnet
      subnet = unit
      for layer_name, template_layer in sorted(subnet.layer_data_templates.items()):
        assert isinstance(template_layer, _TemplateLayer)
        assert issubclass(template_layer.layer_class_type, LayerBase)
        for loss in template_layer.layer_class_type.get_losses(reduce_func=reduce_func, **template_layer.kwargs):
          assert isinstance(loss, LossHolder)
          if layer:
            assert loss.name in subnet.accumulated_losses
            loss = subnet.accumulated_losses[loss.name]
            assert isinstance(loss, LossHolder)
            assert loss.get_layer()
          loss = loss.copy_new_base(network=network, name="%s/%s" % (name, loss.name), reduce_func=reduce_func)
          losses.append(loss)
    return losses

  def get_constraints_value(self):
    """
    :rtype: tf.Tensor
    """
    v = super(RecLayer, self).get_constraints_value()
    from returnn.tf.util.basic import optional_add
    if isinstance(self.cell, _SubnetworkRecCell):
      layers = list(self.cell.net.layers.values())
      if self.cell.input_layers_net:
        layers += list(self.cell.input_layers_net.layers.values())
      if self.cell.output_layers_net:
        layers += list(self.cell.output_layers_net.layers.values())
      for layer in layers:
        v = optional_add(v, layer.get_constraints_value())
    return v

  def _get_cell(self, unit, unit_opts=None):
    """
    :param str|_SubnetworkRecCell|type unit:
    :param None|dict[str] unit_opts:
    :rtype: _SubnetworkRecCell|tensorflow.contrib.rnn.RNNCell|tensorflow.contrib.rnn.FusedRNNCell|TFNativeOp.RecSeqCellOp  # nopep8
    """
    from returnn.tf.util.basic import is_gpu_available_in_session
    rnn_contrib = None
    try:
      # noinspection PyUnresolvedReferences
      from tensorflow.contrib import rnn as rnn_contrib
    except ImportError:
      pass
    import returnn.tf.native_op as tf_native_op
    if isinstance(unit, _SubnetworkRecCell):
      assert unit_opts is None
      unit.set_parent_layer(self)
      return unit
    assert isinstance(unit, (str, type))
    rnn_cell_class = self.get_rnn_cell_class(unit)
    n_hidden = self.output.dim
    if unit_opts is None:
      unit_opts = {}
    if is_gpu_available_in_session():
      try:
        # noinspection PyUnresolvedReferences
        from tensorflow.contrib import cudnn_rnn
      except ImportError:
        pass
      else:
        if issubclass(rnn_cell_class, (cudnn_rnn.CudnnLSTM, cudnn_rnn.CudnnGRU)):
          # noinspection PyArgumentList
          cell = rnn_cell_class(
            num_layers=1, num_units=n_hidden,
            input_mode='linear_input', direction='unidirectional', dropout=0.0, **unit_opts)
          return cell
    if issubclass(rnn_cell_class, tf_native_op.RecSeqCellOp):
      # noinspection PyArgumentList
      cell = rnn_cell_class(
        n_hidden=n_hidden, n_input_dim=self.input_data.dim,
        input_is_sparse=self.input_data.sparse,
        step=self._direction, **unit_opts)
      return cell
    # noinspection PyArgumentList
    cell = rnn_cell_class(n_hidden, **unit_opts)
    base_types = (rnn_cell.RNNCell,)
    if rnn_contrib:
      # noinspection PyUnresolvedReferences
      base_types += (rnn_contrib.FusedRNNCell, rnn_contrib.LSTMBlockWrapper)
    assert isinstance(cell, base_types)  # e.g. BasicLSTMCell
    return cell

  def _get_output_cell(self, cell):
    """
    :param tensorflow.contrib.rnn.RNNCell|tensorflow.contrib.rnn.FusedRNNCell cell:
    :return: output of shape (time, batch, dim)
    :rtype: tf.Tensor
    """
    from tensorflow.python.ops import rnn
    rnn_contrib = None
    try:
      # noinspection PyUnresolvedReferences
      from tensorflow.contrib import rnn as rnn_contrib
    except ImportError:
      pass
    assert self.input_data
    assert not self.input_data.sparse
    in_data, x, seq_len = self._get_input()
    if self._direction == -1:
      x = tf_compat.v1.reverse_sequence(x, seq_lengths=seq_len, batch_dim=1, seq_dim=0)
    if isinstance(cell, BaseRNNCell):
      with tf_compat.v1.variable_scope(tf_compat.v1.get_variable_scope(), initializer=self._fwd_weights_initializer):
        x = cell.get_input_transformed(x)
    if isinstance(cell, rnn_cell.RNNCell):  # e.g. BasicLSTMCell
      if not self.input_data.have_time_axis() or not self.time_dim_tag:
        assert self._direction in [1, None]
        assert self._rec_previous_layer, "%s: assume in loop with input %s, but no rec info" % (self, self.input_data)
        y, final_state = cell(self.input_data.placeholder, self._initial_state)
        in_data = None  # mark that we have not used this
      elif self._unroll:
        assert self._max_seq_len is not None, "specify max_seq_len for unroll"
        # We must get x.shape[0] == self._max_seq_len, so pad it.
        x_shape = x.get_shape().as_list()
        original_len = tf.shape(x)[0]
        # With unrolling, normally we would require max_seq_len >= original_len.
        # Earlier, we just truncated it in that case and filled with zero afterwards,
        # which is bad, as this silently introduces wrong behavior for this case.
        with tf.control_dependencies([
              tf_compat.v1.assert_greater_equal(
                self._max_seq_len, original_len,
                message="required for unroll: max_seq_len >= seq_len")]):
          pad_len = tf.maximum(0, self._max_seq_len - original_len)  # max, in case we want to support truncate later
          x = tf.pad(x, [(0, pad_len), (0, 0), (0, 0)])
        x.set_shape([self._max_seq_len] + x_shape[1:])
        x = tf.unstack(x, axis=0, num=self._max_seq_len)
        y, final_state = rnn.static_rnn(
          cell=cell, dtype=tf.float32, inputs=x, sequence_length=seq_len,
          initial_state=self._initial_state)
        y = tf.stack(y, axis=0)
        y.set_shape([self._max_seq_len, None, self.output.dim])  # (time,batch,ydim)
        # Now, recover the original len.
        y = y[:original_len]
      else:
        # Will get (time,batch,ydim).
        assert self._max_seq_len is None
        y, final_state = rnn.dynamic_rnn(
          cell=cell, inputs=x, time_major=True, sequence_length=seq_len, dtype=tf.float32,
          initial_state=self._initial_state)
      self._last_hidden_state = final_state
    elif rnn_contrib and isinstance(cell, (rnn_contrib.FusedRNNCell, rnn_contrib.LSTMBlockWrapper)):  # noqa # e.g. LSTMBlockFusedCell
      # Will get (time,batch,ydim).
      assert self._max_seq_len is None
      y, final_state = cell(
        inputs=x, sequence_length=seq_len, dtype=tf.float32,
        initial_state=self._initial_state)
      self._last_hidden_state = final_state
    else:
      raise Exception("invalid type: %s" % type(cell))
    if self._direction == -1:
      y = tf_compat.v1.reverse_sequence(y, seq_lengths=seq_len, batch_dim=1, seq_dim=0)
    if in_data:
      y = self._post_proc_output_cell_strict(y, in_data=in_data)
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
    from returnn.tf.util.basic import get_current_var_scope_name
    # noinspection PyUnresolvedReferences
    from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
    assert self._max_seq_len is None
    assert self.input_data
    assert not self.input_data.sparse
    in_data, x, seq_len = self._get_input()
    n_batch = tf.shape(seq_len)[0]
    if self._direction == -1:
      x = tf_compat.v1.reverse_sequence(x, seq_lengths=seq_len, batch_dim=1, seq_dim=0)
    with tf_compat.v1.variable_scope("cudnn"):
      cell.build(x.get_shape())
      num_layers = 1
      # noinspection PyProtectedMember
      rnn_mode = cell._rnn_mode
      param_size = self._get_cudnn_param_size(
        num_units=self.output.dim, input_size=self.input_data.dim, rnn_mode=rnn_mode, num_layers=num_layers)
      # Note: The raw params used during training for the cuDNN op is just a single variable
      # with all params concatenated together.
      # For the checkpoint save/restore, we will use Cudnn*Saveable, which also makes it easier in CPU mode
      # to import the params for another unit like LSTMBlockCell.
      # Also see:
      # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/cudnn_rnn/python/kernel_tests/cudnn_rnn_ops_test.py
      params = cell.kernel
      params.set_shape([param_size])
      if rnn_mode == cudnn_rnn_ops.CUDNN_LSTM:
        fn = cudnn_rnn_ops.CudnnLSTMSaveable
      elif rnn_mode == cudnn_rnn_ops.CUDNN_GRU:
        fn = cudnn_rnn_ops.CudnnGRUSaveable
      elif rnn_mode == cudnn_rnn_ops.CUDNN_RNN_TANH:
        fn = cudnn_rnn_ops.CudnnRNNTanhSaveable
      elif rnn_mode == cudnn_rnn_ops.CUDNN_RNN_RELU:
        fn = cudnn_rnn_ops.CudnnRNNReluSaveable
      else:
        raise ValueError("rnn mode %r" % rnn_mode)
      params_saveable = fn(
        params,
        num_layers=cell.num_layers,
        num_units=cell.num_units,
        input_size=cell.input_size,
        input_mode=cell.input_mode,
        direction=cell.direction,
        scope="%s/params_canonical" % get_current_var_scope_name(),
        name="%s/params_canonical" % get_current_var_scope_name())
      tf_compat.v1.add_to_collection(tf_compat.v1.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
      self.saveable_param_replace[params] = params_saveable
      # It's like a fused cell, i.e. operates on the full sequence.
      input_h = tf.zeros((num_layers, n_batch, self.output.dim), dtype=tf.float32)
      input_c = tf.zeros((num_layers, n_batch, self.output.dim), dtype=tf.float32)
      y, _ = cell(x, initial_state=(input_h, input_c))
    if self._direction == -1:
      y = tf_compat.v1.reverse_sequence(y, seq_lengths=seq_len, batch_dim=1, seq_dim=0)
    y = self._post_proc_output_cell_strict(y, in_data=in_data)  # noqa
    return y  # noqa

  def _get_output_native_rec_op(self, cell):
    """
    :param returnn.tf.native_op.RecSeqCellOp cell:
    :return: output of shape (time, batch, dim)
    :rtype: tf.Tensor
    """
    from returnn.tf.util.basic import dot, sequence_mask_time_major, directed, to_int32_64, set_param_axes_split_info

    # Some error checking.
    from returnn.tf import native_op
    if isinstance(cell, native_op.NativeLstmCell):
      if self._initial_state is not None:
        raise Exception(
          "%s: NativeLstm (v1) has broken (only partial) support for initial_state. "
          "See here for details: https://github.com/rwth-i6/returnn/issues/813. "
          "Use 'NativeLstm2' instead." % self)

    assert self._max_seq_len is None
    assert self.input_data
    in_data, x, seq_len = self._get_input()
    if self._input_projection:
      if cell.does_input_projection:
        # The cell get's x as-is. It will internally does the matrix mult and add the bias.
        pass
      else:
        weights = tf_compat.v1.get_variable(
          name="W", shape=(self.input_data.dim, cell.n_input_dim), dtype=tf.float32,
          initializer=self._fwd_weights_initializer)
        if self.input_data.sparse:
          x = tf.nn.embedding_lookup(weights, to_int32_64(x))
        else:
          x = dot(x, weights)
        b = tf_compat.v1.get_variable(
          name="b", shape=(cell.n_input_dim,), dtype=tf.float32, initializer=self._bias_initializer)
        if len(cell.n_input_dim_parts) > 1:
          set_param_axes_split_info(weights, [[self.input_data.dim], cell.n_input_dim_parts])
          set_param_axes_split_info(b, [cell.n_input_dim_parts])
        x += b
    else:
      assert not cell.does_input_projection
      assert not self.input_data.sparse
      assert self.input_data.dim == cell.n_input_dim
    index = sequence_mask_time_major(seq_len)
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
    y = self._post_proc_output_cell_strict(y, in_data=in_data)
    return y

  def _get_output_subnet_unit(self, cell):
    """
    :param _SubnetworkRecCell cell:
    :return: output of shape (time, batch, dim)
    :rtype: tf.Tensor
    """
    output = cell.get_output()
    self._last_hidden_state = cell
    return output

  def get_last_hidden_state(self, key):
    """
    :param str|int|None key:
    :rtype: tf.Tensor
    """
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

  def get_sub_layer(self, layer_name):
    """
    :param str layer_name: name of the sub_layer (right part of '/' separated path)
    :return: the sub_layer addressed in layer_name or None if no sub_layer exists
    :rtype: LayerBase|None
    """
    if isinstance(self.cell, _SubnetworkRecCell):
      # try to find layer_name in cell:
      return self.cell.get_layer_from_outside(layer_name)
    return None

  def get_sub_networks(self):
    """
    :rtype: list[returnn.tf.network.TFNetwork]
    """
    if isinstance(self.cell, _SubnetworkRecCell):
      return self.cell.get_sub_networks()
    return []

  def get_sub_layers(self):
    """
    :rtype: list[LayerBase]
    """
    if isinstance(self.cell, _SubnetworkRecCell):
      return self.cell.get_all_layers_shallow()
    return []


class _SubnetworkRecCell(object):
  """
  This class is used by :class:`RecLayer` to implement
  the generic subnetwork logic inside the recurrency.
  """

  _debug_out = None  # set to list to enable

  def __init__(self, net_dict, source_data, time_dim_tag, rec_layer_name, parent_net, parent_get_layer):
    """
    :param dict[str,dict[str]] net_dict: dict for the subnetwork, layer name -> layer dict
    :param Data|None source_data: usually concatenated input from the rec-layer. template
    :param Dim time_dim_tag:
    :param str rec_layer_name:
    :param returnn.tf.network.TFNetwork parent_net:
    :param ((str) -> LayerBase) parent_get_layer: for template construction
    """
    from returnn.tf.util.basic import safe_deep_copy
    self.parent_rec_layer = None  # type: typing.Optional[RecLayer]
    self.parent_net = parent_net
    self.net_dict = safe_deep_copy(net_dict)
    from returnn.tf.network import TFNetwork, ExternData, LossHolder
    from returnn.tf.util.data import ControlFlowContext
    control_flow_ctx = ControlFlowContext(
      kind=ControlFlowContext.Types.Loop, outer_ctx=parent_net.get_control_flow_ctx())
    control_flow_ctx.loop_spatial_dim = time_dim_tag
    self.net = TFNetwork(
      name="%s/%s(rec-subnet)" % (parent_net.name, rec_layer_name),
      extern_data=ExternData(),
      train_flag=parent_net.train_flag,
      search_flag=parent_net.search_flag,
      eval_flag=False,
      inside_rec_time_dim=time_dim_tag,
      control_flow_ctx=control_flow_ctx,
      absolute_name_prefix="%s%s/" % (parent_net.get_absolute_name_prefix(), rec_layer_name),
      parent_net=parent_net)
    self.net.is_root_in_ctx = True
    self.net.layers_desc.update(self.net_dict)
    self.source_data = source_data
    if source_data:
      self.net.extern_data.data["source"] = (
        source_data.copy_template_excluding_time_dim().copy_template_set_ctx(control_flow_ctx))
    self.time_dim_tag = time_dim_tag
    self._time_dim_tags = {time_dim_tag}  # type: typing.Set[Dim]
    if source_data:
      # Maybe the input has a different time dim tag, but we still have new dynamic length here
      # because of custom endings ("end" layer).
      self._time_dim_tags.add(source_data.get_time_dim_tag())
    for key, data in parent_net.extern_data.data.items():
      if key in self.net.extern_data.data or data.time_dim_axis is None:
        continue  # Don't overwrite existing, e.g. "source".
      # These are just templates. You can use them as possible targets for dimension information,
      # but not as actual sources or targets.
      # Note: We maybe should check data.is_same_time_dim()...
      self.net.extern_data.data[key] = data.copy_template_excluding_time_dim().copy_template_set_ctx(control_flow_ctx)
    self.layer_data_templates = {}  # type: typing.Dict[str,_TemplateLayer]
    self.prev_layers_needed = set()  # type: typing.Set[str]
    self.prev_layer_templates = {}  # type: typing.Dict[str,_TemplateLayer]
    self._template_construction_exceptions = None  # type: typing.Optional[typing.List[str]]
    self._construct_template(parent_get_layer=parent_get_layer)
    self._last_frames = {}  # type: typing.Dict[str,Data]
    self._initial_outputs = None  # type: typing.Optional[typing.Dict[str,tf.Tensor]]
    self._initial_extra_outputs = None  # type: typing.Optional[typing.Dict[str,typing.Dict[str,typing.Union[tf.Tensor,typing.Tuple[tf.Tensor,...]]]]]  # nopep8

    # input_layers_moved_out, output_layers_moved_out and layers_in_loop include (used) sub-layers as separate
    # entries, this way in- and outputting them to the loop via TensorArrays will be handled just as for normal
    # layers.
    self.input_layers_moved_out = []  # type: typing.List[str]
    self.output_layers_moved_out = []  # type: typing.List[str]
    self.layers_in_loop = None   # type: typing.Optional[typing.List[str]]
    self.input_layers_net = None  # type: typing.Optional[TFNetwork]
    self.output_layers_net = None  # type: typing.Optional[TFNetwork]
    self.final_acc_tas_dict = None  # type: typing.Optional[typing.Dict[str, tf.TensorArray]]
    self.get_final_rec_vars = None
    self.accumulated_losses = {}  # type: typing.Dict[str,LossHolder]

  def __repr__(self):
    return "<%s %r>" % (self.__class__.__name__, self.net.name)

  def set_parent_layer(self, parent_rec_layer):
    """
    Only called from RecLayer._get_cell to define itself as parent

    :param RecLayer parent_rec_layer:
    """
    assert parent_rec_layer.network is self.parent_net
    parent_rec_layer.cell = self
    self.parent_rec_layer = parent_rec_layer
    self.net.parent_layer = parent_rec_layer

  def _construct_template(self, parent_get_layer):
    """
    Without creating any computation graph, create TemplateLayer instances.
    Need it for shape/meta information as well as dependency graph in advance.
    It will init self.layer_data_templates and self.prev_layers_needed.

    :param ((str) -> LayerBase) parent_get_layer:
    """
    import sys
    from returnn.util import better_exchook
    from pprint import pformat
    from collections import OrderedDict
    from returnn.util.basic import StringIO, BehaviorVersion
    from returnn.tf.network import NetworkConstructionDependencyLoopException, DataNotFound
    # The stack trace is not so interesting for these exceptions.
    skip_stack_trace_exception_types = (
      NetworkConstructionDependencyLoopException,)

    # These Exceptions always indicate incorrect construction, so fail directly instead of collecting them
    fail_directly_exception_types = (DataNotFound, LayerNotFound, BehaviorVersion.RequirementNotSatisfied)

    # noinspection PyShadowingNames
    def _parent_get_layer(layer_name):
      """
      :param str layer_name:
      :rtype: LayerBase
      """
      try:
        return parent_get_layer(layer_name)
      except Exception as exc:
        exc._is_exc_from_parent_get_layer = True
        raise

    class CollectedException:
      """
      Collected exception information.
      """
      def __init__(self, text, exception):
        """
        :param str text: formatted exception with optional stack trace
        :param Exception exception:
        """
        self.text = text
        self.exception = exception

    class ConstructCtx:
      """
      Closure.
      """
      layers = []  # type: typing.List[_TemplateLayer]  # stack of layers
      most_recent = None  # type: typing.Optional[typing.List[_TemplateLayer]]  # most recent stack
      partially_finished = []  # type: typing.List[_TemplateLayer]
      collected_exceptions = OrderedDict()  # type: OrderedDict[object,CollectedException]  # key custom below
      recent_exception = None  # type: Exception

      # noinspection PyShadowingNames
      @classmethod
      def collect_exception(cls, layer_name):
        """
        Collect most recent exception.
        Pretty generic exception handling but anything could happen.
        We don't do any output by default, as this could be very spammy,
        but we collect the traceback, in case we get some other error later.
        Then go on with the next get_layer.

        :param str layer_name:
        """
        exc_type, value, tb = sys.exc_info()
        if getattr(value, "_is_exc_from_parent_get_layer", False):
          raise
        exc_last_frame = list(better_exchook.iter_traceback(tb))[-1]
        exc_key = (exc_last_frame.f_code.co_filename, exc_last_frame.f_lineno, exc_last_frame.f_code.co_name)
        if exc_key not in cls.collected_exceptions:
          if isinstance(value, skip_stack_trace_exception_types):
            color = better_exchook.Color()
            cls.collected_exceptions[exc_key] = CollectedException(
              text="%s\n%s: %s\n" % (
                color("EXCEPTION while constructing layer %r" % layer_name, color.fg_colors[1], bold=True),
                color(exc_type.__name__, color.fg_colors[1]),
                str(value)),
              exception=value)
          else:
            if isinstance(value, fail_directly_exception_types):
              raise
            out = StringIO()
            better_exchook.better_exchook(exc_type, value, tb, file=out)
            cls.collected_exceptions[exc_key] = CollectedException(text=out.getvalue(), exception=value)
        if not cls.recent_exception or not isinstance(value, skip_stack_trace_exception_types):
          cls.recent_exception = value

      @classmethod
      def fail(cls):
        """
        Fail by reraising the most recent relevant exception,
        or raising some new exception.
        """
        if cls.recent_exception:
          raise cls.recent_exception
        raise Exception("Failed to construct Rec subnet template")

    class RecSubnetGetTemplateLayer:
      """
      Helper class to provide the ``get_layer`` function with specific properties.
      """
      # noinspection PyMethodParameters
      def __init__(lself,
                   allow_uninitialized_template=False,
                   iterative_testing=True, reconstruct=False,
                   parent=None, parent_name=None):
        """
        :param bool allow_uninitialized_template: whether an uninitialized template layer can be returned
        :param bool iterative_testing: whether we should iterate through multiple get_layer variants
        :param bool reconstruct: if layer exists and is initialized, do not return it but reconstruct it.
          It could have been initialized with incorrect sources (marked as partially finished),
          and thus the data output might be wrong.
        :param GetLayer|None parent:
        :param str|None parent_name:
        """
        lself.allow_uninitialized_template = allow_uninitialized_template
        if parent:
          assert isinstance(parent, RecSubnetGetTemplateLayer)
        lself.parent = parent
        lself.parent_name = parent_name
        lself.iterative_testing = iterative_testing
        lself.reconstruct = reconstruct
        lself.got_uninitialized_deps_count = 0

      # noinspection PyMethodParameters
      def __repr__(lself):
        parent_names = []
        parent = lself
        while parent and parent.parent:
          parent_names.insert(0, parent.parent_name or "?")
          parent = parent.parent
        return (
          "<RecLayer construct template GetLayer>("
          "allow_uninitialized_template %r, "
          "parents %r)") % (
                 lself.allow_uninitialized_template,
                 " <- ".join(parent_names) or None)

      def _add_uninitialized_count(self):
        getter = self
        while getter:
          assert isinstance(getter, RecSubnetGetTemplateLayer)
          getter.got_uninitialized_deps_count += 1
          getter = getter.parent

      def reset(self):
        """
        Reset.
        """
        self.got_uninitialized_deps_count = 0

      def construct(self, layer_name_):
        """
        Note: Different from just __call__: We reset first.
        Also, we catch exceptions.
        The layer should be in partially_finished in any case afterwards,
        and we make sure later that we finally manage to construct them all.

        :param str layer_name_:
        """
        assert not self.parent
        self.reset()
        # noinspection PyBroadException
        try:
          self.__call__(layer_name_)
        except Exception:
          ConstructCtx.collect_exception(layer_name=layer_name_)

      # noinspection PyMethodParameters
      def add_templated_layer(lself, name, layer_class, **layer_desc):
        """
        This is used instead of self.net.add_layer because we don't want to add
        the layers at this point, we just want to construct the template layers
        and store inside self.layer_data_templates.

        :param str name:
        :param type[LayerBase]|LayerBase layer_class:
        :param layer_desc:
        :rtype: LayerBase
        """
        # _TemplateLayer already created in get_templated_layer.
        assert name in self.layer_data_templates, (
          "%s: Unexpected layer name %r. Templates:\n%s" % (self, name, pformat(self.layer_data_templates)))
        layer_ = self.layer_data_templates[name]
        layer_desc = layer_desc.copy()
        layer_desc["name"] = name
        layer_desc["network"] = self.net
        old_layer_kwargs = layer_.kwargs
        # set it now already for better debugging
        layer_.layer_class_type = layer_class
        layer_.kwargs = layer_desc.copy()
        if "output" not in layer_.kwargs:
          if old_layer_kwargs and "output" in old_layer_kwargs:
            # First copy old output. Maybe the get_out_data_from_opts raises an exception,
            # and we don't want this to be unset.
            layer_.kwargs["output"] = old_layer_kwargs["output"]
          layer_.kwargs["output"] = layer_class.get_out_data_from_opts(**layer_desc)
        # noinspection PyBroadException
        try:
          layer_.kwargs["output"] = layer_class.fixup_out_data(**layer_.kwargs)
        except Exception:  # most likely VerifyOutShapeException
          if lself.got_uninitialized_deps_count > 0:
            pass  # ignore in this case
          else:
            raise
        layer_.kwargs["output"].sanity_check(ignore_placeholder=True)  # placeholder might be overwritten later
        layer_.init(layer_class=layer_class, **layer_.kwargs)
        if layer_.need_last:
          self.prev_layers_needed.add(name)
        if layer_ in ConstructCtx.partially_finished:
          if lself.got_uninitialized_deps_count == 0:  # in this case, we safely know that it is finished
            ConstructCtx.partially_finished.remove(layer_)
        return layer_

      # noinspection PyMethodParameters
      def __call__(lself, name, is_prev_time_frame=False):
        """
        This is the get_layer function implementation.

        :param str name: layer name
        :param bool is_prev_time_frame: layer of prev frame ("prev:...")
        :return: layer, or None
        :rtype: LayerBase
        """
        _name = name
        if name.startswith("prev:"):
          assert not is_prev_time_frame, "%s: only prev frame can be accessed" % self
          name = name[len("prev:"):]
          self.prev_layers_needed.add(name)
          layer_ = lself.__call__(name, is_prev_time_frame=True)
          if name in self.layer_data_templates:
            assert isinstance(layer_, _TemplateLayer)
            if layer_.is_initialized:
              if layer_ not in ConstructCtx.partially_finished:  # it might not be final
                layer_ = self.get_prev_template_layer(name)
              else:
                lself._add_uninitialized_count()
                layer_ = layer_.copy_as_prev_time_frame()
            else:
              lself._add_uninitialized_count()
          return layer_
        earlier_layer_output = None  # type: typing.Optional[Data]
        if name in self.layer_data_templates:
          layer_ = self.layer_data_templates[name]
          if ConstructCtx.layers:
            ConstructCtx.layers[-1].add_dependency(layer_, is_prev_time_frame=is_prev_time_frame)
          if lself.allow_uninitialized_template:
            if not layer_.is_initialized or layer_ in ConstructCtx.partially_finished:
              lself._add_uninitialized_count()
            return layer_
          if not lself.reconstruct and layer_.is_initialized:
            if layer_ in ConstructCtx.partially_finished:
              lself._add_uninitialized_count()
            return layer_
          if layer_.is_initialized:
            earlier_layer_output = layer_.output
        if name.startswith("base:"):
          assert not is_prev_time_frame
          layer_ = _parent_get_layer(name[len("base:"):])
          if ConstructCtx.layers:
            ConstructCtx.layers[-1].add_dependency(layer_, is_prev_time_frame=False)
          return layer_
        # Need to create layer instance here now to not run into recursive loops.
        # We will extend it later in add_templated_layer().
        if name in self.layer_data_templates:  # might exist already
          layer_ = self.layer_data_templates[name]
        else:
          layer_ = _TemplateLayer(
            name=name, network=self.net, cell=self,
            construct_stack=ConstructCtx.layers[-1] if ConstructCtx.layers else None)
          self.layer_data_templates[name] = layer_
        res_layer = None
        if ConstructCtx.layers:
          ConstructCtx.layers[-1].add_dependency(layer_, is_prev_time_frame=is_prev_time_frame)
        if layer_ not in ConstructCtx.partially_finished:
          # Add it early. We want to catch all possible source of exceptions/errors, via:
          # * layer_class.transform_config_dict (via construct_layer)
          # * layer_class.get_out_data_from_opts (via add_templated_layer)
          ConstructCtx.partially_finished.append(layer_)
        if lself.allow_uninitialized_template:
          lself._add_uninitialized_count()
          return layer_
        default_success = False  # whether construction was successful with default_get_layer
        ConstructCtx.layers.append(layer_)
        try:
          initial_output_layer = None
          # If not initialized yet and this is still the dummy data template,
          # see if we can figure out a petter template.
          if not layer_.is_initialized and layer_.output.batch_ndim == 0:
            if name in self.net_dict:  # only true for the root net
              layer_dict = self.net_dict[name]
              initial_output = layer_dict.get("initial_output")
              if isinstance(initial_output, str) and initial_output.startswith("base:"):
                initial_output_layer = _parent_get_layer(initial_output[len("base:"):])
                layer_.output = initial_output_layer.output.copy_template(name="%s_output" % name)

          ConstructCtx.most_recent = list(ConstructCtx.layers)
          default_get_layer = RecSubnetGetTemplateLayer(parent=lself, parent_name=_name)
          last_get_layer = default_get_layer
          # noinspection PyBroadException
          try:
            res_layer = self.net.construct_layer(
              net_dict=self.net_dict, name=name,
              get_layer=default_get_layer, add_layer=default_get_layer.add_templated_layer)
            default_success = True
          except Exception:
            ConstructCtx.collect_exception(layer_name=name)
            if layer_.is_initialized:
              # Return anyway. This will be resolved later.
              # Also, this is important to not keep trying again and again potentially for a very long time.
              # https://github.com/rwth-i6/returnn/issues/1127
              pass
            elif lself.iterative_testing or initial_output_layer:
              pass  # cover this below
            else:
              raise

          if not layer_.is_initialized and lself.iterative_testing:
            shallow_get_layer = RecSubnetGetTemplateLayer(
              allow_uninitialized_template=True, parent=lself, parent_name=_name)
            last_get_layer = shallow_get_layer
            # noinspection PyBroadException
            try:
              self.net.construct_layer(
                net_dict=self.net_dict, name=name,
                get_layer=shallow_get_layer, add_layer=shallow_get_layer.add_templated_layer)
            except Exception:
              ConstructCtx.collect_exception(layer_name=name)
              if not initial_output_layer:
                raise

          if initial_output_layer and layer_ in ConstructCtx.partially_finished:
            # We can trust the initial output, and even ignore whatever else we got from the construct_layer above.
            # The construct_layer is just to setup the deps and kwargs.
            assert layer_.kwargs
            layer_.kwargs["output"] = initial_output_layer.output.copy_template(name="%s_output" % name)
            last_get_layer.add_templated_layer(
              layer_class=layer_.layer_class_type, **layer_.kwargs)

          if res_layer and res_layer is not layer_:
            assert isinstance(res_layer, _TemplateLayer), "unexpected %s, via %s" % (res_layer, layer_)
            assert res_layer.is_initialized, "unexpected %s, via %s" % (res_layer, layer_)
            # Copy relevant information over to layer_ instance.
            lself.add_templated_layer(
              layer_class=res_layer.layer_class_type, **res_layer.kwargs)

        except LayerNotFound as exc:
          if exc.network is self.net and exc.layer_name == name:
            assert not layer_.is_initialized
            # Clean up any references.
            layer_ = self.layer_data_templates.pop(name)
            ConstructCtx.partially_finished.remove(layer_)
            for other in list(layer_.used_by):
              if isinstance(other, _TemplateLayer):
                other.remove_dependency(layer_, is_prev_time_frame=is_prev_time_frame)
          raise

        finally:
          assert ConstructCtx.layers[-1] is layer_, "invalid stack %r, expected top layer %r" % (
            ConstructCtx.layers, layer_)
          ConstructCtx.layers.pop(-1)

          if layer_.is_initialized:
            if not layer_.output.size_placeholder and earlier_layer_output and earlier_layer_output.size_placeholder:
              # E.g. during reconstruct, but based on other partially finished / incomplete sources.
              # However, maybe we got useful dim tags / size placeholders from a previous (partial) construction.
              # Copy if it matches.
              # Do this even if there was an exception, but with new partial construction.
              if earlier_layer_output.matches_var_dim_pattern(layer_.output):
                layer_.output.size_placeholder = earlier_layer_output.size_placeholder.copy()

        # It was constructed now.
        assert layer_.is_initialized
        if not default_success:
          lself._add_uninitialized_count()
        return layer_

    get_templated_layer = RecSubnetGetTemplateLayer()

    try:
      assert not self.layer_data_templates, "do not call this multiple times"
      get_templated_layer.construct("output")
      if "output" not in self.layer_data_templates:
        raise ConstructCtx.recent_exception
      assert not ConstructCtx.layers

      if "end" in self.net_dict:  # used to specify ending of a sequence
        get_templated_layer.construct("end")

      for layer_name, layer in sorted(self.net_dict.items()):
        if self.parent_net.eval_flag and layer.get("loss"):  # only collect losses if we need them
          get_templated_layer.construct(layer_name)
      for layer_name, layer in sorted(self.net_dict.items()):
        if layer.get("is_output_layer") or layer.get("need_last"):
          get_templated_layer.construct(layer_name)

      # Because of the logic to lazily init deps, or some of the kwargs sources partially None,
      # we might have some layers still uninitialized, or should reinit with correct sources.
      # Note that it is hard to get the order right how we do this, as there are circular dependencies.
      # Thus, we might need several loops.
      # Note that it is still not guaranteed that this will finish. We stop if there was no change anymore.
      # We also put an additional limit, which might be hit due to other bugs.
      loop_limit = len(ConstructCtx.partially_finished)
      direct_get_layer = RecSubnetGetTemplateLayer(iterative_testing=False, reconstruct=True)
      while ConstructCtx.partially_finished:
        old_len = len(ConstructCtx.partially_finished)
        recent_changes = []
        for layer in ConstructCtx.partially_finished:
          old_output = layer.output
          direct_get_layer.construct(layer.name)
          if layer.output.get_compare_key() != old_output.get_compare_key():
            recent_changes.append(("layer %r:" % layer.name, old_output, "vs", layer.output))
        if len(ConstructCtx.partially_finished) < old_len:
          # Ok, this is some real progress.
          continue
        if len(ConstructCtx.partially_finished) >= old_len and not recent_changes:
          # No changes anymore. There is no real point in continuing. Just break.
          if not all([layer.is_initialized for layer in ConstructCtx.partially_finished]):
            print(
              "Failed to initialize layers:\n" +
              "".join(["  %s\n" % layer for layer in ConstructCtx.partially_finished if not layer.is_initialized]) +
              "Check the further debug output for the partial construction and other exceptions.")
            ConstructCtx.fail()
          break
        loop_limit -= 1
        if loop_limit < 0:
          print(
            ("We keep iterating over the network template construction.\n"
             "We have these partially finished layers:\n%s\n"
             "And these finished layers:\n%s\n"
             "And these recent changes in the last loop iteration:\n%s") % (
              pformat(ConstructCtx.partially_finished),
              pformat([
                layer for _, layer in sorted(self.layer_data_templates.items())
                if layer not in ConstructCtx.partially_finished]),
              pformat(recent_changes)))
          ConstructCtx.fail()

      # If we got so far, cleanup collected exceptions a bit.
      for key, exc_info in list(ConstructCtx.collected_exceptions.items()):
        if isinstance(exc_info.exception, NetworkConstructionDependencyLoopException):
          del ConstructCtx.collected_exceptions[key]
      # And keep the remaining ones for potential later reports.
      self._template_construction_exceptions = [s.text for s in ConstructCtx.collected_exceptions.values()]

    except Exception:
      print("%r: exception constructing template network (for deps and data shapes)" % self)
      from pprint import pprint
      print("Most recent construction stack:")
      if ConstructCtx.most_recent:
        for layer in ConstructCtx.most_recent:
          assert isinstance(layer, _TemplateLayer)
          print("%r, kwargs:" % (layer,))
          pprint(layer.kwargs)
      else:
        print(ConstructCtx.most_recent)
      print("Template network so far:")
      pprint(self.layer_data_templates)
      print("Collected (unique) exceptions during template construction:")
      print("(Note that many of these can be ignored, or are expected.)")
      for s in ConstructCtx.collected_exceptions.values():
        print(s.text)
      raise

  def _add_template_layer(self, layer_name, layer_dict):
    """
    Use this for simple helpers, after the main template net is already created.
    This does not expect layer creation exceptions,
    and expects that all dependencies are already created,
    and all dependencies are other layers inside our subnet.

    :param str layer_name:
    :param dict[str] layer_dict: not yet transformed
    :rtype: _TemplateLayer
    """
    from returnn.tf.network import get_layer_class
    assert layer_name not in self.layer_data_templates
    self.net_dict[layer_name] = layer_dict
    # We replicate the _construct_template logic here, but simplified,
    # i.e. not expecting exceptions, and expecting that all dep layers are created.
    layer = _TemplateLayer(name=layer_name, network=self.net, cell=self)
    layer_dict = layer_dict.copy()
    layer_class_name = layer_dict.pop("class")
    layer_class = get_layer_class(layer_class_name)
    layer_dict.update(dict(name=layer_name, _name=layer_name, network=self.net, _network=self.net))
    layer_class.transform_config_dict(
      layer_dict, network=self.net, get_layer=lambda _name: self.layer_data_templates[_name])
    layer_dict["output"] = layer_class.get_out_data_from_opts(**layer_dict)
    layer_dict["output"] = layer_class.fixup_out_data(**layer_dict)
    layer.init(layer_class=layer_class, **layer_dict)
    self.layer_data_templates[layer_name] = layer
    return layer

  def _handle_construct_exception(self, description, exception):
    """
    :param str description:
    :param Exception exception:
    """
    from returnn.tf.network import LayerNotFound, DataNotFound
    if isinstance(exception, (DataNotFound, LayerNotFound)):
      return  # pass on. this is maybe caught elsewhere. also other template exceptions are probably not relevant.
    if getattr(exception, "_RETURNN_handled_construct_exception", None):
      return  # already handled before
    out = log.v2
    print(file=out)
    print("ERROR: Got exception during %s:" % description, file=out)
    print("%s: %s" % (type(exception).__name__, exception), file=out)
    print(file=out)
    print("Template network (check out types / shapes):", file=out)
    for key, value in self.layer_data_templates.items():
      print("%s: %s" % (key, value), file=out)
    if self._template_construction_exceptions:
      print(file=out)
      print("Collected (unique) exceptions during template construction:", file=out)
      print("(Note that many of these can be ignored, or are expected.)", file=out)
      for s in self._template_construction_exceptions:
        print(file=out)
        print(s, file=out, end="")
      # Don't print twice.
      self._template_construction_exceptions = None
    print(file=out)
    exception._RETURNN_handled_construct_exception = True

  def _get_parent_layer(self, layer_name):
    """
    :param str layer_name: without "base:" prefix
    :rtype: LayerBase
    """
    return self.parent_net.get_layer(layer_name)

  def _construct(self, prev_outputs, prev_extra, i, data=None,
                 inputs_moved_out_tas=None, needed_outputs=("output",)):
    """
    This is called from within the `tf.while_loop` of the :class:`RecLayer`,
    to construct the subnetwork, which is performed step by step.

    :param dict[str,tf.Tensor] prev_outputs: outputs of the layers from the previous step
    :param dict[str,dict[str,tf.Tensor]] prev_extra: extra output / hidden states of the previous step for layers
    :param tf.Tensor i: loop counter. scalar, int32, current step (time)
    :param dict[str,tf.Tensor] data: All data needed from outside of the loop. Possible keys are 'source'
        (for the input of the recurrent layer) and the keys in parent_net.extern_data, which notably include
        the target of the recurrent layer, usually called 'classes'.
    :param dict[str,tf.TensorArray]|None inputs_moved_out_tas:
    :param set[str] needed_outputs: layers where we need outputs
    """
    from returnn.tf.network import TFNetwork
    from .base import WrappedInternalLayer
    for key in data:
      self.net.extern_data.data[key].placeholder = data[key]
    for data_key, data in self.net.extern_data.data.items():
      if data_key not in self.net.used_data_keys:
        continue
      if data.placeholder is None:
        raise Exception("rec layer %r subnet data key %r is not set" % (self.parent_rec_layer.name, data_key))

    prev_layers = {}  # type: typing.Dict[str,_TemplateLayer]
    for name in set(list(prev_outputs.keys()) + list(prev_extra.keys())):
      if "prev:%s" % name in self.net.layers:
        continue
      try:
        self.net.layers["prev:%s" % name] = prev_layers[name] = self.layer_data_templates[name].copy_as_prev_time_frame(
          prev_output=prev_outputs.get(name, None),
          rec_vars_prev_outputs=prev_extra.get(name, None))
      except Exception as exc:  # Some sanity check might have failed or so.
        self._handle_construct_exception(description="in-loop init of prev layer %r" % name, exception=exc)
        raise

    inputs_moved_out = {}  # type: typing.Dict[str,WrappedInternalLayer]

    # noinspection PyShadowingNames
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
      if layer_name not in inputs_moved_out_tas:
        assert not layer.output.mark_same_time(self._time_dim_tags), (
          "%s does not expect to have matching time dim to %s" % (layer, self.parent_rec_layer))
        assert name != "output" and not prev, "Time dim does not match: RecLayer %s (%r) vs sub layer %s (%r)." % (
          self.parent_rec_layer, self.parent_rec_layer.output.get_time_dim_tag(),
          layer, layer.output.get_time_dim_tag())
        return layer
      output = layer.output.copy_template_excluding_time_dim().copy_template_set_ctx(self.net.control_flow_ctx)
      with tf.name_scope("%s_moved_input" % name.replace(":", "_")):
        if prev:
          output.placeholder = tf.cond(
            tf.equal(i, 0),
            lambda: self._get_init_output(layer_name),
            lambda: inputs_moved_out_tas[layer_name].read(i - 1))
        else:
          output.placeholder = inputs_moved_out_tas[layer_name].read(i)
        output.sanity_check()
      layer = self.net.add_layer(
        name=name, output=output, layer_class=WrappedInternalLayer, base_layer=layer)
      inputs_moved_out[name] = layer
      return layer

    # noinspection PyShadowingNames
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
        layer = self._get_parent_layer(name[len("base:"):])
        return layer
      if name in self.input_layers_moved_out:
        return get_input_moved_out(name)
      if name in self.output_layers_moved_out:
        # Will be constructed later.
        # This should not be used recursively, because we checked that nothing depends on it,
        # thus it should not be a problem to return None.
        return None
      # noinspection PyBroadException
      try:
        layer = self.net.construct_layer(self.net_dict, name=name, get_layer=get_layer)
        # Some layers are buggy to determine the right beam size at template construction time.
        # Usually this is because they ignore some of the dependencies in get_out_data_from_opts.
        # If that is the case, this will likely crash at some later point with mismatching shape.
        # Do an explicit check here now, to easier localize such problems.
        layer_template = self.layer_data_templates[name]
        layer_choices = layer.get_search_choices()
        if not layer.search_choices and layer_choices:
          assert (layer.output.beam == layer_template.output.beam and
                  layer_choices.beam_size == layer.output.beam.beam_size == layer_template.output.beam.beam_size), (
            "Layer %r has buggy search choices resolution." % layer,
            self.net.debug_search_choices(layer) or "see search choices debug output")
        if name == "end":
          # Special logic for the end layer, to always logical_or to the prev:end layer.
          assert layer.output.shape == (), "%s: 'end' layer %r unexpected shape" % (self.parent_rec_layer, layer)
          prev_end_layer = self.net.layers["prev:end"]
          choices = layer.get_search_choices()
          if choices:
            prev_end_layer = choices.translate_to_this_search_beam(prev_end_layer)
          with tf.name_scope("end_flag"):
            layer.output.placeholder = tf.logical_or(prev_end_layer.output.placeholder, layer.output.placeholder)
        return layer
      except Exception as exc:
        self._handle_construct_exception(description="in-loop construction of layer %r" % name, exception=exc)
        raise

    # Go through needed_outputs, e.g. "output".
    # And prev_layers_needed because they might not be resolved otherwise.
    for layer_name in sorted(needed_outputs) + sorted(self.prev_layers_needed):
      if layer_name in self.input_layers_moved_out + self.output_layers_moved_out:
        continue
      layer = get_layer(layer_name)
      if '/' not in layer_name:  # sub-layers are not in self.net
        assert layer_name in self.net.layers
      if layer_name in prev_layers:
        prev_layer = prev_layers[layer_name]
        assert layer.output.batch_shape == prev_layer.output.batch_shape
        assert layer.output.batch_dim_axis == prev_layer.output.batch_dim_axis
        assert sorted(layer.output.size_placeholder.keys()) == sorted(prev_layer.output.size_placeholder.keys())
        for i in range(len(layer.output.size_placeholder)):
          assert layer.output.get_size_dim_tag(i) == prev_layer.output.get_size_dim_tag(i)

  def get_prev_template_layer(self, layer_name):
    """
    :param str layer_name: without "prev:"
    :return: prev template layers. makes sure that we don't recreate them
    :rtype: _TemplateLayer
    """
    if ("prev:%s" % layer_name) in self.net.layers:  # this is even better
      return self.net.layers["prev:%s" % layer_name]
    if layer_name in self.prev_layer_templates:
      return self.prev_layer_templates[layer_name]
    template_layer = self.layer_data_templates[layer_name]
    layer = template_layer.copy_as_prev_time_frame()
    self.prev_layer_templates[layer_name] = layer
    return layer

  def get_layer_from_outside(self, layer_name):
    """
    :param str layer_name: name of the sub_layer (addressed by '/' separated path)
    :return: the sub_layer addressed in layer_name or None if no sub_layer exists
    :rtype: LayerBase|None
    """
    if self.output_layers_net and layer_name in self.output_layers_net.layers:
      return self.output_layers_net.layers[layer_name]
    elif self.input_layers_net and layer_name in self.input_layers_net.layers:
      return self.input_layers_net.layers[layer_name]
    elif self.net and layer_name in self.net.layers:
      raise Exception(
        "%r: Cannot get layer %r from outside, because it is only available inside the recurrent loop. \
         Add 'is_output_layer':True to the layer options." % (self.parent_rec_layer, layer_name))
    return None

  def get_layer_last_frame(self, layer_name):
    """
    :param str layer_name:
    :return: the output from the last loop frame. compatible to layer templates. masking correctly taken into account
    :rtype: Data
    """
    return self._last_frames[layer_name]

  def get_sub_networks(self):
    """
    :rtype: list[returnn.tf.network.TFNetwork]
    """
    # Note that the order matters, to resolve duplicate names (by absolute name)
    # in get_all_layers_deep.
    return [net for net in [self.input_layers_net, self.net, self.output_layers_net] if net]

  def get_all_layers_shallow(self):
    """
    :rtype: list[LayerBase]
    """
    layer_set = set()
    layers = []
    for net in self.get_sub_networks():
      if not layers:
        layers += net.get_all_layers_shallow()
        layer_set.update(layers)
      else:
        for layer in net.get_all_layers_shallow():
          if layer not in layer_set:
            layers.append(layer)
            layer_set.add(layer)
    return layers

  def _get_init_output(self, name, batch_dim=None):
    """
    :param str name: layer name
    :param tf.Tensor|None batch_dim:
    :rtype: tf.Tensor
    """
    template_layer = self.layer_data_templates[name]
    cl = template_layer.layer_class_type
    assert issubclass(cl, LayerBase)
    # noinspection PyProtectedMember
    with reuse_name_scope(self.parent_rec_layer._rec_scope):
      with cl.cls_setup_scope(**template_layer.kwargs):
        # noinspection PyBroadException
        try:
          if batch_dim is None:
            batch_dim = template_layer.get_batch_dim()
          if name == "end" and template_layer.kwargs.get("initial_output", None) is None:
            # Special case for the 'end' layer.
            from returnn.tf.util.basic import constant_with_shape
            return constant_with_shape(False, shape=[batch_dim], name="initial_end")
          return cl.get_rec_initial_output(
            batch_dim=batch_dim, rec_layer=self.parent_rec_layer, **self.layer_data_templates[name].kwargs)
        except Exception as exc:
          self._handle_construct_exception(
            description="initial-output construction of layer %r" % name, exception=exc)
          raise

  def _get_init_extra_outputs(self, name):
    """
    :param str name: layer name
    :rtype: dict[str,tf.Tensor]
    """
    template_layer = self.layer_data_templates[name]
    cl = template_layer.layer_class_type
    assert issubclass(cl, LayerBase)
    # noinspection PyProtectedMember
    with reuse_name_scope(self.parent_rec_layer._rec_scope):
      with cl.cls_setup_scope(**template_layer.kwargs):
        # noinspection PyBroadException
        try:
          batch_dim = template_layer.get_batch_dim()
          d = cl.get_rec_initial_extra_outputs(
            batch_dim=batch_dim, rec_layer=self.parent_rec_layer, **self.layer_data_templates[name].kwargs)
        except Exception as exc:
          self._handle_construct_exception(
            description="initial-extra-output construction of layer %r" % name, exception=exc)
          raise
    return d

  def _check_output_template_shape(self):
    output_template = self.layer_data_templates["output"]
    assert output_template.output.dim == self.parent_rec_layer.output.dim
    assert self.parent_rec_layer.output.time_dim_axis == 0
    assert not output_template.output.has_axis(self.time_dim_tag)
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
    from returnn.util.basic import sorted_values_from_dict
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
    from returnn.tf.util.basic import nested_get_shapes
    init_rec_extra_shapes = nested_get_shapes(self._initial_extra_outputs)
    for name, shapes in init_rec_extra_shapes.items():
      # See also _get_init_extra_outputs.
      template_layer = self.layer_data_templates[name]
      cl = template_layer.layer_class_type
      d = cl.get_rec_initial_extra_outputs_shape_invariants(
        rec_layer=self.parent_rec_layer, **self.layer_data_templates[name].kwargs)
      for k, shape in d.items():
        assert k in shapes
        # Not merge but replace because we intentionally want to allow relaxation.
        shapes[k] = shape
    from returnn.util.basic import sorted_values_from_dict
    init_outputs_flat = sorted_values_from_dict(init_out_shapes)
    init_extra_flat = [sorted_values_from_dict(v) for (k, v) in sorted(init_rec_extra_shapes.items())]
    return init_outputs_flat, init_extra_flat

  def get_layer_rec_var_from_loop_vars(self, loop_vars, layer_name, final_frame=False, seq_len=None):
    """
    :param (list[tf.Tensor],list[tf.Tensor]) loop_vars: loop_vars like in self.get_next_loop_vars()
    :param str layer_name:
    :param bool final_frame:
    :param tf.Tensor seq_len: if final frame, this is the seq len, shape (batch,)
    :return: layer rec_vars_outputs
    :rtype: dict[str,tf.Tensor]
    """
    prev_outputs_flat, prev_extra_flat = loop_vars
    assert len(prev_outputs_flat) == len(self._initial_outputs)
    assert len(prev_extra_flat) == len(self._initial_extra_outputs)
    from returnn.util.basic import dict_zip
    prev_extra = {
      k: dict_zip(sorted(self._initial_extra_outputs[k]), v)
      for (k, v) in zip(sorted(self._initial_extra_outputs), prev_extra_flat)}
    rec_vars_outputs = prev_extra[layer_name]
    if final_frame:
      if layer_name in self.net.layers:
        rec_vars_outputs = self.net.layers[layer_name].post_process_final_rec_vars_outputs(
          rec_vars_outputs, seq_len=seq_len)
    return rec_vars_outputs

  def _assign_last_frames_from_loop_vars(self, loop_vars):
    """
    :param (list[tf.Tensor],list[tf.Tensor]) loop_vars: loop_vars like in self.get_next_loop_vars()
    """
    final_outputs_flat, final_extra_flat = loop_vars
    assert len(final_outputs_flat) == len(self._initial_outputs)
    assert len(final_extra_flat) == len(self._initial_extra_outputs)
    for k, v in zip(sorted(self._initial_outputs), final_outputs_flat):
      layer_template = self.layer_data_templates[k]
      if not layer_template.need_last:
        continue
      out = layer_template.output.copy_template()
      out.placeholder = v
      assert k not in self._last_frames
      self._last_frames[k] = out

  def _assign_last_frames_from_optimized(self):
    for k in self.prev_layers_needed:
      if k in self.output_layers_moved_out:
        layer = self.output_layers_net.layers[k]
      elif k in self.input_layers_moved_out:
        layer = self.input_layers_net.layers[k]
      else:
        continue
      assert isinstance(layer, LayerBase)
      if not layer.need_last:
        continue
      with tf.name_scope(layer.tf_scope_name):
        out = layer.output.copy_as_batch_major().copy_with_time_dim_axis(1)  # [B,T,...]
        time_dim = out.dim_tags[1]
        if time_dim.dyn_size_ext:
          indices = time_dim.dyn_size_ext.copy()
        else:
          indices = Data.from_tensor(tf_util.get_shape_dim(out.placeholder, 0))
        indices.placeholder = indices.placeholder - 1
        v = tf.gather(
          out.placeholder,
          indices=tf.maximum(
            indices.copy_compatible_to(
              Data("dummy", dim_tags=out.dim_tags[:1], dtype="int32"), unbroadcast=True).placeholder, 0),
          batch_dims=1, axis=1)  # [B,...]
        layer_template = self.layer_data_templates[k]
        out = layer_template.output.copy_template()
        v = tf.where(
          tf.less(indices.copy_compatible_to(out, check_sparse=False, check_dtype=False).placeholder, 0),
          self._get_init_output(k), v)
        out.placeholder = v
        assert k not in self._last_frames
        self._last_frames[k] = out

  def get_parent_deps(self):
    """
    :return: list of dependencies to the parent network
    :rtype: list[LayerBase]
    """
    ls = []

    def maybe_add(layer_):
      """
      :param LayerBase layer_:
      """
      # Usually dep.network is self.cell.net but it could reference to our own net,
      # e.g. if this is an attention layer like
      # {"class": "dot_attention", "base": "base:encoder", ...}.
      if layer_.network is self.parent_net:
        if layer_ not in ls:
          ls.append(layer_)

    layers = self.net.layers
    if not layers:  # happens only during initialization
      layers = self.layer_data_templates
    for _, layer in sorted(layers.items()):
      assert isinstance(layer, LayerBase)
      maybe_add(layer)
      if isinstance(layer, _TemplateLayer):
        for dep in layer.dependencies:  # if it is uninitialized, need to use this
          maybe_add(dep)
      else:
        for dep in layer.get_dep_layers():
          maybe_add(dep)
    return ls

  def _while_loop(self, cond, body, loop_vars, shape_invariants):
    """
    :param function cond:
    :param function body:
    :param T loop_vars:
    :param S shape_invariants:
    :rtype: T
    """
    return tf.while_loop(
      cond=cond,
      body=body,
      loop_vars=loop_vars,
      shape_invariants=shape_invariants,
      back_prop=self.parent_rec_layer.back_prop)

  class OutputToAccumulate:
    """
    Helper class to hold information about some tensor which we are going to accumulate in a TensorArray
    from inside of the recurrent loop.
    """

    # noinspection PyShadowingNames
    def __init__(self, name, get, dtype=None, element_shape=None, same_shape_every_frame=None, data=None):
      """
      :param str name:
      :param ()->(Data|tf.Tensor|None) get:
      :param tf.DType|str dtype:
      :param tuple[int|None] element_shape:
      :param bool same_shape_every_frame:
      :param Data|None data:
      """
      from returnn.tf.util.basic import get_valid_scope_name_from_str
      self.name = name
      self.tf_scope_name = get_valid_scope_name_from_str(name.replace("/", "_"))
      self.data = data
      if dtype is None:
        assert data
        dtype = data.dtype
      self.dtype = dtype
      if element_shape is None:
        assert data
        element_shape = data.batch_shape
      self.element_shape = element_shape
      if same_shape_every_frame is None:
        if data:
          same_shape_every_frame = not data.have_varying_shape_in_ctx()
        else:
          same_shape_every_frame = True
      self.same_shape_every_frame = same_shape_every_frame
      self.get = get
      self.get_returned_none = None  # type: typing.Optional[bool]

    def __repr__(self):
      return "%s{%r}" % (self.__class__.__name__, self.name)

    def write_to_tensor_array(self, ta, index):
      """
      :param tf.TensorArray ta:
      :param tf.Tensor index:
      :return: new ta
      :rtype: tf.TensorArray
      """
      assert self.get_returned_none is None
      value = self.get()
      if value is None:
        self.get_returned_none = True
        return ta
      else:
        if isinstance(value, tf.Tensor):
          pass
        elif isinstance(value, Data):
          assert self.data
          assert value.dtype == self.data.dtype
          assert value.batch_shape == self.data.batch_shape
          assert value.have_varying_shape_in_ctx() == self.data.have_varying_shape_in_ctx()
          value = value.placeholder
        else:
          raise TypeError("OutputToAccumulate.get: expected tf.Tensor or Data but got %r" % type(value))
        self.get_returned_none = False
        return ta.write(index=index, value=value, name="%s_acc_ta_write" % self.tf_scope_name)

    def get_final_tensor_array(self, ta):
      """
      :param tf.TensorArray ta:
      :return: ta if we wrote to it, otherwise None
      :rtype: tf.TensorArray|None
      """
      assert self.get_returned_none is not None
      if self.get_returned_none:
        return None
      if not self.same_shape_every_frame:
        ta = self._make_padded_tensor_array(ta)
      return ta

    def _make_padded_tensor_array(self, ta):
      """
      :param tf.TensorArray ta:
      :rtype: tf.TensorArray
      """
      assert not self.same_shape_every_frame
      # First get the max shape of each element. Then create a new TensorArray which can hold all elements.
      # Because we use clear_after_read in ta, even to get the max shape, we have to create a new TensorArray.
      assert self.data
      size = ta.size()
      buffer_ta = tf.TensorArray(
        name="acc_ta_infer_max_shape_%s" % self.tf_scope_name,
        dtype=self.dtype,
        element_shape=tf.TensorShape(self.element_shape),
        size=size,
        clear_after_read=True,
        infer_shape=False)

      def _body_infer_max_shape(i, max_shape_, new_ta_):
        """
        :param tf.Tensor i: scalar
        :param tf.Tensor max_shape_:
        :param tf.TensorArray new_ta_:
        """
        elem = ta.read(i)
        max_shape_ = tf.maximum(max_shape_, tf.shape(elem))
        new_ta_ = new_ta_.write(i, elem)
        return i + 1, max_shape_, new_ta_

      max_shape = tf.convert_to_tensor([d if d else 0 for d in self.data.batch_shape], dtype=tf.int32)
      _, max_shape, buffer_ta = tf.while_loop(
        cond=lambda i, *_args: tf.less(i, size),
        body=_body_infer_max_shape,
        loop_vars=(0, max_shape, buffer_ta))

      # Now again create a new TensorArray.
      new_ta_padded = tf.TensorArray(
        name="acc_ta_pad_max_shape_%s" % self.tf_scope_name,
        dtype=self.dtype,
        element_shape=tf.TensorShape(self.element_shape),
        size=size,
        clear_after_read=True,
        infer_shape=True)

      def _body_pad_max_shape(i, new_ta_):
        """
        :param tf.Tensor i: scalar
        :param tf.TensorArray new_ta_:
        """
        from returnn.tf.util.basic import get_shape
        elem = buffer_ta.read(i)
        elem_shape = get_shape(elem)
        pad_values = [(0, max_shape[a] - elem_shape[a]) for a in range(len(elem_shape))]
        elem_padded = tf.pad(elem, pad_values)
        new_ta_ = new_ta_.write(i, elem_padded)
        return i + 1, new_ta_

      _, new_ta_padded = tf.while_loop(
        cond=lambda i, *_args: tf.less(i, size),
        body=_body_pad_max_shape,
        loop_vars=(0, new_ta_padded))
      return new_ta_padded

  def get_output(self):
    """
    :return: output of shape (time, batch, dim), search choices
    :rtype: tf.Tensor
    """
    from returnn.tf.util.basic import check_input_dim, tensor_array_stack, Dim, get_valid_scope_name_from_str
    from returnn.util.basic import BehaviorVersion
    assert self.parent_rec_layer
    rec_layer = self.parent_rec_layer

    # The template network is already constructed at this point, but nothing else.
    self._check_output_template_shape()
    output_template = self.layer_data_templates["output"]
    output_template_search_choices = output_template.get_search_choices()
    if output_template_search_choices and output_template_search_choices.owner.network is not self.net:
      # We are only interested in search choices happening inside this rec layer.
      output_template_search_choices = None

    # dict to collect all data that will be fed from outside of the rec_layer. If present, this includes
    # the input ('source') and the target, but maybe also other additional extern data that is used inside the subnet.
    data_tensor_arrays = {}  # dict[str,tf.TensorArray]

    time_dim_tag = None
    with tf.name_scope("subnet_base"):
      batch = rec_layer.get_batch_info()
      input_beam = None  # type: typing.Optional[SearchBeam]
      if rec_layer.input_data:
        with tf.name_scope("source_tensor_array"):
          # noinspection PyProtectedMember
          source_data, source, input_seq_len = rec_layer._get_input(strict=False)  # (time,batch,..,dim)
          source_shape = tf.shape(source, name="source_shape")
          source_ta = tf.TensorArray(
            name="source_ta",
            dtype=rec_layer.input_data.dtype,
            element_shape=tf.TensorShape(source_data.copy_template_excluding_time_dim().batch_shape),
            size=source_shape[0],
            infer_shape=True)
          source_ta = source_ta.unstack(source, name="source_ta_unstack")
          data_tensor_arrays["source"] = source_ta
        input_search_choices = rec_layer.network.get_search_choices(sources=rec_layer.sources)
        if input_search_choices:
          assert rec_layer.input_data.beam.beam_size == input_search_choices.search_choices.beam_size
          input_beam = rec_layer.input_data.beam
        elif rec_layer.input_data.beam:
          input_beam = rec_layer.input_data.beam
      else:
        input_seq_len = None
        if rec_layer.target and rec_layer.network.eval_flag:
          # noinspection PyProtectedMember
          target_data = rec_layer._get_target_value(
            target=rec_layer.target, mark_data_key_as_used=False)
          input_beam = target_data.beam
      fixed_seq_len = None
      if rec_layer.output.size_placeholder and not output_template_search_choices:
        # See LayerBase._post_init_output(). could be set via target or size_target...
        # This should only be the case in training.
        fixed_seq_len = rec_layer.output.size_placeholder[0]
      elif rec_layer.size_target:  # if this is set, always use it
        # noinspection PyProtectedMember
        tag = rec_layer._get_target_value(
          target=rec_layer.size_target, mark_data_key_as_used=True).get_time_dim_tag()
        tag = tag.get_for_batch_ctx(batch, rec_layer.output.control_flow_ctx)
        if tag.dyn_size_ext and tag.dyn_size_ext.placeholder is not None:
          fixed_seq_len = tag.dyn_size
      elif rec_layer.time_dim_tag:
        tag = rec_layer.time_dim_tag.get_for_batch_ctx(batch, rec_layer.output.control_flow_ctx)
        if tag.dyn_size_ext and tag.dyn_size_ext.placeholder is not None:
          fixed_seq_len = tag.dyn_size
      if fixed_seq_len is None and "end" not in self.layer_data_templates:
        # If 'end' layer is not existing, the length must be defined.
        # In some cases (training with given target) we know the target sequence length.
        # Otherwise, by convention, it is defined by the input length
        # (assuming that there is an input which we iterate over).
        assert input_seq_len is not None, "length is not defined. provide an 'end' layer"
        fixed_seq_len = input_seq_len
      if fixed_seq_len is not None:
        time_dim_tag = Dim.get_tag_from_size_tensor(fixed_seq_len)
        assert time_dim_tag == self.time_dim_tag
        if fixed_seq_len.get_shape().ndims > 0:
          assert fixed_seq_len.get_shape().ndims == 1
          with tf.name_scope("check_seq_len_batch_size"):
            # It should already be adapted to our batch.
            fixed_seq_len = check_input_dim(fixed_seq_len, axis=0, dim=batch.dim)
            if time_dim_tag:
              time_dim_tag.set_tag_on_size_tensor(fixed_seq_len, batch=time_dim_tag.batch, same_as_before=True)
        max_seq_len = tf.reduce_max(fixed_seq_len, name="max_seq_len")
        have_known_seq_len = True
      else:
        assert "end" in self.layer_data_templates, "length not defined, provide 'end' layer"
        max_seq_len = None
        have_known_seq_len = False
        # self.time_dim_tag (via transform_config_dict) should match the logic in here.
        # Ie. for this case the dyn size should not be set, as it will be dynamic later on via "end".
        assert self.time_dim_tag.dyn_size is None
      if time_dim_tag:
        self.time_dim_tag.declare_same_as(time_dim_tag)
      else:
        time_dim_tag = self.time_dim_tag

      common_data_len = None  # used to check whether all extern data have same length
      used_keys = self.net.used_data_keys.copy()
      for key in sorted(used_keys):
        data = rec_layer.network.get_extern_data(key, mark_data_key_as_used=True)
        # This could be another dim tag from the rec time dim tag.
        self._time_dim_tags.add(data.get_time_dim_tag())
        data_placeholder = data.get_placeholder_as_time_major()
        with tf.name_scope("check_data_len"):
          data_len = tf.shape(data_placeholder)[0]
          if common_data_len is None:
            # Check for first key if input length matches data length
            if input_seq_len is not None:
              with tf.control_dependencies(
                  [tf_compat.v1.assert_equal(
                    tf.reduce_max(input_seq_len), data_len,
                    ["RecLayer %r with sources %r:" % (rec_layer.name, rec_layer.sources),
                     " The length of the sources (", tf.reduce_max(input_seq_len),
                     ") differ from the length of the target ", key, "(", data_len, ")."])]):
                data_len = tf.identity(data_len)
            if fixed_seq_len is not None:
              with tf.control_dependencies(
                  [tf_compat.v1.assert_equal(
                    tf.reduce_max(fixed_seq_len), data_len,
                    ["RecLayer %r:" % (rec_layer.get_absolute_name(),),
                     " The predefined length (", tf.reduce_max(fixed_seq_len),
                     ") differs from the length of the target ", key, "(", data_len, ")."])]):
                data_len = tf.identity(data_len)
            common_data_len = data_len
          else:
            # Check from second key on if data length is equal for all external data
            with tf.control_dependencies([
              tf_compat.v1.assert_equal(
                common_data_len, data_len,
                ["RecLayer %r:" % rec_layer.name, " The length of all targets (%s) " % ", ".join(used_keys),
                 " has to be the same. Found length ", data_len, " for %s, which does not match length " % key,
                 common_data_len, " of the other data."])]):
              data_len = tf.identity(data_len)
        data_ta = tf.TensorArray(
          name=key + "_ta",
          dtype=data.dtype,
          element_shape=tf.TensorShape(data.copy_template_excluding_time_dim().batch_shape),
          size=data_len,
          infer_shape=True)
        data_ta = data_ta.unstack(data_placeholder, name="{}_ta_unstack".format(key))
        data_tensor_arrays[key] = data_ta
        if max_seq_len is None:
          max_seq_len = common_data_len

      # Note: tf.while_loop() will not give us all intermediate outputs, but we want them.
      # tf.scan() would do that but tf.scan() will loop over some input sequence -
      # however, that would not work because the input sequence is not fixed initially.
      # So, similar to tf.scan() does it, we collect all intermediate values.

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

      outputs_to_accumulate = []  # type: typing.List[_SubnetworkRecCell.OutputToAccumulate]
      needed_outputs = {"output"}  # names. these are needed somewhere
      extra_output_layers = set()  # names. will create accumulated output layer in any case for these

      # noinspection PyShadowingNames
      def add_output_to_acc(layer_name):
        """
        :param str layer_name:
        """
        name_ = "output_%s" % layer_name
        if any([(out.name == name_) for out in outputs_to_accumulate]):
          return
        template_layer = self.layer_data_templates[layer_name]
        outputs_to_accumulate.append(_SubnetworkRecCell.OutputToAccumulate(
          name=name_,
          data=template_layer.output,
          get=lambda: self.net.get_layer(layer_name).output))

      for name, template in self.layer_data_templates.items():
        if template.is_output_layer():
          needed_outputs.add(name)
          extra_output_layers.add(name)
        if template.need_last:
          needed_outputs.add(name)

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
      output_beam = None  # type: typing.Optional[SearchBeam]
      collected_choices = []  # type: typing.List[str]  # layer names
      for layer in self.layer_data_templates.values():
        assert isinstance(layer, _TemplateLayer)
        if layer.search_choices:
          collected_choices += [layer.name]

          # noinspection PyShadowingNames
          def get_choices_getter(name):
            """
            :param str name:
            :rtype: ()->tf.Tensor|None
            """
            def get_choice_source_batches():
              """
              :rtype: tf.Tensor|None
              """
              layer = self.net.layers[name]
              return layer.search_choices.src_beams
            return get_choice_source_batches

          outputs_to_accumulate.append(
            _SubnetworkRecCell.OutputToAccumulate(
              name="choice_%s" % layer.name,
              dtype=tf.int32,
              element_shape=(None, layer.search_choices.beam_size),  # (batch, beam)
              get=get_choices_getter(layer.name)))

      if collected_choices:
        output_beam = self.layer_data_templates["output"].output.beam
        # Note: output_beam_size can be None, if output itself does not depend on any choice,
        # which might be uncommon, but is valid.

      if not have_known_seq_len:
        assert "end" in self.layer_data_templates, "You need to have an 'end' layer in your rec subnet."
        end_template = self.layer_data_templates["end"]
        needed_outputs.add("end")
        assert tf.as_dtype(end_template.output.dtype) is tf.bool
        assert end_template.output.batch_shape == (None,)  # (batch*beam,)
      else:
        assert have_known_seq_len, (
          "You need to have an 'end' layer in your rec subnet if the generated seq len is unknown.")

      # noinspection PyProtectedMember
      if self.parent_rec_layer._optimize_move_layers_out:
        self._move_outside_loop(needed_outputs=needed_outputs)
      else:
        self.layers_in_loop = sorted(self.layer_data_templates.keys())

      accumulated_loop_losses = {}  # name -> loss holder. only losses inside the loop
      if layer_names_with_losses:
        # noinspection PyShadowingNames
        def make_get_loss_in_loop_frame(loss, layer_name, return_error=False, return_loss=False):
          """
          :param LossHolder loss:
          :param str layer_name:
          :param bool return_error:
          :param bool return_loss:
          :rtype: ()->tf.Tensor
          """
          from returnn.tf.network import LossHolder
          assert isinstance(loss, LossHolder)

          def get_loop_loss():
            """
            :rtype: tf.Tensor|None
            """
            layer = self.net.layers[layer_name]
            loss.init(layer)
            if return_loss:
              value = loss.get_loss_value()
            elif return_error:
              value = loss.get_error_value()
            else:
              assert False, "return_error or return_loss"
            if return_error and value is None:
              return None
            assert isinstance(value, tf.Tensor), "layer %r loss %r %s invalid" % (
              layer, loss, "loss_value" if return_loss else "error_value")
            assert value.get_shape().ndims >= 1
            if value.get_shape().ndims > 1:  # e.g. BinaryCrossEntropy
              value = tf.reduce_sum(value, axis=list(range(1, value.get_shape().ndims)))
            value.set_shape(tf.TensorShape((None,)))  # (batch,)
            return value

          return get_loop_loss

        from returnn.tf.util.basic import identity
        for layer_name in layer_names_with_losses:
          if layer_name not in self.layers_in_loop:
            continue  # will get loss out of them below
          layer = self.layer_data_templates[layer_name]
          assert issubclass(layer.layer_class_type, LayerBase)
          # Using the identity reduce_func is a bit hacky
          # but we do not want to reduce the loss to a scalar in the loop
          # but get it as shape (batch,).
          # This should work with all current implementations
          # but might need some redesign later.
          for loss in layer.layer_class_type.get_losses(reduce_func=identity, **layer.kwargs):
            assert loss.name not in accumulated_loop_losses, "layer %r loss name %r not unique" % (layer, loss.name)
            accumulated_loop_losses[loss.name] = loss
            outputs_to_accumulate.append(_SubnetworkRecCell.OutputToAccumulate(
              name="loss_%s" % loss.name,
              dtype=tf.float32,
              element_shape=(None,),  # (batch,)
              get=make_get_loss_in_loop_frame(loss=loss, layer_name=layer_name, return_loss=True)))
            outputs_to_accumulate.append(_SubnetworkRecCell.OutputToAccumulate(
              name="error_%s" % loss.name,
              dtype=tf.float32,
              element_shape=(None,),  # (batch,)
              get=make_get_loss_in_loop_frame(loss=loss, layer_name=layer_name, return_error=True)))

      if "output" in self.layers_in_loop:
        add_output_to_acc("output")

      # if a layer declares it is a output, we should save the values as well
      for name in extra_output_layers:
        if name in self.layers_in_loop:
          add_output_to_acc(name)

      if rec_layer.debug:
        if layer_name in self.layers_in_loop:
          outputs_to_accumulate.append(_SubnetworkRecCell.OutputToAccumulate(
            name="debug_output_%s" % layer_name,
            data=self.layer_data_templates[layer_name].output,
            get=lambda name_=layer_name: self.net.get_layer(name_).output))

      # Maybe some of the moved-out output-layers depend on data inside the loop,
      # so we should accumulate it to have access to it.
      for layer_name in self.output_layers_moved_out:
        for dep in self.layer_data_templates[layer_name].dependencies:
          if dep.name not in self.layers_in_loop:
            continue
          # Dependency is inside the loop, and we are using it, so we need to accumulate its output.
          add_output_to_acc(dep.name)
          needed_outputs.add(dep.name)

      in_loop_ctx_dim_tags = set()
      for layer_name in self.layers_in_loop:
        if layer_name in list(needed_outputs):
          layer = self.layer_data_templates[layer_name]
          for tag in layer.output.dim_tags:
            if tag.control_flow_ctx == self.net.control_flow_ctx:
              if tag not in in_loop_ctx_dim_tags:
                in_loop_ctx_dim_tags.add(tag)
                # The helper layer name does not matter except for debugging and it should not clash with other layers.
                # The helper layer name matters only on the sense that it must come sorted before other extra layers,
                # such that we construct it first in _construct_output_layers_moved_out.
                helper_layer_name = ":dyn-tag-accum:%i:%s" % (len(in_loop_ctx_dim_tags), layer_name)
                helper_layer_dict = {"class": "length", "from": layer_name, "axis": tag}
                self._add_template_layer(helper_layer_name, helper_layer_dict)
                add_output_to_acc(helper_layer_name)
                needed_outputs.add(helper_layer_name)
                extra_output_layers.add(helper_layer_name)

      # Tensor arrays for any layers which were moved out.
      input_layers_moved_out_tas = {}
      if self.input_layers_moved_out:
        with tf.name_scope("input_layers_moved_out"):
          self._construct_input_layers_moved_out()
          for layer_name in self.input_layers_moved_out:
            # Create only Tensor arrays for those which we use inside the loop.
            if not self._input_layer_used_inside_loop(layer_name):
              continue
            layer = self.input_layers_net.get_layer(layer_name)
            assert isinstance(layer, LayerBase)
            if layer_name == "output":
              assert layer.output.have_time_axis()
              assert layer.output.get_time_dim_tag() in self._time_dim_tags
            # Only unroll if that is the same time dim.
            if not layer.output.mark_same_time(self._time_dim_tags):
              continue
            assert max_seq_len is not None
            inp_ta = tf.TensorArray(
              name="%s_ta" % layer_name,
              dtype=self.layer_data_templates[layer_name].output.dtype,
              element_shape=self.layer_data_templates[layer_name].output.batch_shape,
              size=layer.output.time_dimension(),
              infer_shape=True)
            with tf.control_dependencies([
                  tf.Assert(tf.greater_equal(
                    layer.output.time_dimension(), max_seq_len),
                    ["input TA unstack", str(layer.output), "shape", tf.shape(layer.output.placeholder),
                     "seq len", layer.output.get_sequence_lengths(), "do not match",
                     "max seq len", max_seq_len])]):
              inp_ta = inp_ta.unstack(
                layer.output.get_placeholder_as_time_major(),
                name="%s_ta_unstack" % layer_name)
            input_layers_moved_out_tas[layer_name] = inp_ta

      # Create a tensor array to store the intermediate values for each step i, e.g. of shape (batch, dim).
      init_acc_tas = [
        tf.TensorArray(
          name="acc_ta_%s" % out.tf_scope_name,
          dtype=out.dtype,
          element_shape=tf.TensorShape(out.element_shape),
          size=min_loop_len,
          dynamic_size=True,  # we will automatically grow it when needed
          clear_after_read=not out.name.startswith("choice_"),
          infer_shape=out.same_shape_every_frame)
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
      # noinspection PyProtectedMember
      with reuse_name_scope(rec_layer._rec_scope), reuse_name_scope("while_loop_body"):
        step_info_i = i
        # noinspection PyProtectedMember
        if self.parent_rec_layer._use_global_rec_step_offset:
          from returnn.tf.util.basic import global_tensor
          step_info_i += global_tensor(
            lambda: tf_compat.v1.placeholder(tf.int32, (), name="global_rec_step_offset"),
            name="global_rec_step_offset")
        rec_step_info = dict(i=step_info_i, prev_end_flag=None, seq_lens=fixed_seq_len)
        prev_end_flag, prev_dyn_seq_len, prev_end_layer = None, None, None
        if seq_len_info:
          prev_end_flag, prev_dyn_seq_len = seq_len_info
          rec_step_info["prev_end_flag"] = prev_end_flag
          prev_end_layer = self.layer_data_templates["end"].copy_as_prev_time_frame(prev_output=prev_end_flag)
          self.net.layers["prev:end"] = prev_end_layer
          rec_step_info["prev_end_layer"] = prev_end_layer
        self.net.set_rec_step_info(**rec_step_info)
        # get next loop vars (net_vars)
        from returnn.tf.util.basic import identity_op_nested
        from returnn.util.basic import sorted_values_from_dict, dict_zip
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
        data_ = {
          key_: ta.read(i, name="{}_ta_read".format(key_)) for key_, ta in data_tensor_arrays.items()}
        # noinspection PyProtectedMember
        with reuse_name_scope(self.parent_rec_layer._rec_scope):
          self._construct(
            prev_outputs=prev_outputs, prev_extra=prev_extra,
            i=i,
            data=data_,
            inputs_moved_out_tas=input_layers_moved_out_tas,
            needed_outputs=needed_outputs)

        transformed_cache = {}  # type: typing.Dict[LayerBase,LayerBase]  # layer -> layer

        # noinspection PyShadowingNames
        def maybe_transform(layer):
          """
          This will be available in the next loop frame as the "prev:..." layer.
          If the current search choices are already from the prev frame, select beams such that we end up
          in the current frame.
          E.g. let's say the layer "s" has the choices from "prev:output".
          Then "prev:s" (current "s" in the next frame)
          will also have the choices from "prev:output" (current "prev:output" in the next frame).
          This is because there is no "prev:prev:output".

          :param LayerBase layer:
          :rtype: LayerBase
          """
          if layer in transformed_cache:
            return transformed_cache[layer]
          assert not RecLayer.is_prev_step_layer(layer)  # this layer is from current frame
          if not layer.get_search_choices():
            return layer

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

        # Handle need_last.
        for k in sorted(self._initial_outputs):
          layer = self.net.layers[k]
          if not layer.need_last:
            continue
          # We need to make sure that the output is correct also for the seqs which are already ended.
          prev_layer = self.net.layers["prev:" + k]
          # Last frame corresponds to the frame seq_len - 1.
          if seq_len_info:
            # With include_eos=False, when "end" layer is True <=> we are behind the last frame.
            # With include_eos=True, when "prev:end" layer is True <=> we are behind the last frame.
            rel_end_layer = self.net.layers["prev:end"] if rec_layer.include_eos else self.net.layers["end"]
          else:
            assert fixed_seq_len is not None and time_dim_tag and time_dim_tag.dyn_size_ext
            end_flag_data = time_dim_tag.dyn_size_ext.copy_template()
            end_flag_data.dtype = "bool"
            end_flag_data.placeholder = tf.greater_equal(
              # Without include_eos, end_flag=True happens first in frame seq_lens.
              # With include_eos, end_flag=True happens first in frame seq_lens - 1.
              i, (fixed_seq_len - 1) if rec_layer.include_eos else fixed_seq_len)
            from returnn.tf.layers.basic import InternalLayer
            rel_end_layer = InternalLayer(name="rel_end_layer", network=self.net, output=end_flag_data)
          choices = layer.get_search_choices()
          if choices:
            prev_layer, rel_end_layer = choices.translate_to_this_search_beam([prev_layer, rel_end_layer])
          from returnn.tf.util.basic import where_bc
          layer.output.placeholder = where_bc(
            condition=rel_end_layer.output.copy_compatible_to(
              layer.output, check_sparse=False, check_dtype=False).placeholder,
            x=prev_layer.output.placeholder, y=layer.output.placeholder)

        if seq_len_info is not None:
          end_layer = maybe_transform(self.net.layers["end"])
          assert end_layer.output.shape == (), "end layer %r unexpected shape" % end_layer
          end_flag = end_layer.output.placeholder  # we handled the logical_or(prev_end...) in construct get_layer
          end_flag.set_shape([None])
          choices = end_layer.get_search_choices()
          if choices:
            from .basic import SelectSearchSourcesLayer
            prev_end_layer = choices.translate_to_this_search_beam(prev_end_layer)
            assert isinstance(prev_end_layer, SelectSearchSourcesLayer), (
              "unexpected search choices: cur end %r, prev end %r" % (choices, prev_end_layer.get_search_choices()))
            prev_end_flag = prev_end_layer.output.placeholder
            with tf.name_scope("dyn_seq_len"):
              prev_dyn_seq_len = prev_end_layer.transform_func(prev_dyn_seq_len)
          with tf.name_scope("dyn_seq_len"):
            dyn_seq_len = prev_dyn_seq_len + tf.where(
              prev_end_flag if rec_layer.include_eos else end_flag,
              constant_with_shape(0, shape=tf.shape(end_flag)),
              constant_with_shape(1, shape=tf.shape(end_flag)))  # (batch * beam,)
            seq_len_info = (end_flag, dyn_seq_len)

        outputs_flat = [
          maybe_transform(self.net.layers[k]).output.copy_compatible_to(
            self.layer_data_templates[k].output).placeholder
          for k in sorted(self._initial_outputs)]
        extra_flat = []
        for k, v in sorted(self._initial_extra_outputs.items()):
          layer = maybe_transform(self.net.layers[k])
          assert set(layer.rec_vars_outputs.keys()) == set(v.keys())
          extra_flat.append(sorted_values_from_dict(layer.rec_vars_outputs))
        net_vars = (outputs_flat, extra_flat)

        assert len(acc_tas) == len(outputs_to_accumulate)
        acc_tas = [
          out.write_to_tensor_array(acc_ta, index=i)
          for (acc_ta, out) in zip(acc_tas, outputs_to_accumulate)]
        next_i = tf.add(i, 1, name="next_i")
        res = (next_i, net_vars, acc_tas)
        if seq_len_info is not None:
          res += (seq_len_info,)
        if self._debug_out is not None:
          from returnn.tf.util.basic import identity_with_debug_log
          args = {"step": i}
          args.update({"%s.output" % k: v.output.placeholder for (k, v) in self.net.layers.items()})
          for k in self._initial_extra_outputs:
            args.update({"%s.extra.%s" % (k, k2): v for (k2, v) in self.net.layers[k].rec_vars_outputs.items()})
            args.update({"prev:%s.extra.%s" % (k, k2): v for (k2, v) in prev_extra[k].items()})
          res = (identity_with_debug_log(out=self._debug_out, x=res[0], args=args),) + res[1:]
        return res

    # noinspection PyUnusedLocal
    def cond(i, net_vars, acc_tas, seq_len_info=None, allow_inf_max_len=False):
      """
      :param tf.Tensor i: loop counter, scalar
      :param net_vars: the accumulator values. see also self.get_init_loop_vars()
      :param list[tf.TensorArray] acc_tas: the output accumulator TensorArray
      :param (tf.Tensor,tf.Tensor)|None seq_len_info: tuple (end_flag, seq_len)
      :param bool allow_inf_max_len:
      :return: True -> we should run the current loop-iteration, False -> stop loop
      :rtype: tf.Tensor
      """
      with tf.name_scope("loop_cond"):
        from returnn.tf.util.basic import opt_logical_and
        res = True
        # noinspection PyProtectedMember
        if max_seq_len is not None:
          res = opt_logical_and(res, tf.less(i, max_seq_len, name="i_less_max_seq_len"))
        # Only consider the user 'max_seq_len' option if we don't know the real max_seq_len.
        # This is the old behavior. Maybe this might change at some point.
        elif isinstance(rec_layer._max_seq_len, (int, tf.Tensor)):
          # noinspection PyProtectedMember
          res = opt_logical_and(res, tf.less(i, rec_layer._max_seq_len, name="i_less_max_seq_len"))
        else:
          # noinspection PyProtectedMember
          assert rec_layer._max_seq_len is None, "%r: unsupported max_seq_len %r" % (rec_layer, rec_layer._max_seq_len)
        # Check not considering seq_len_info because the dynamics of the network can also lead
        # to an infinite loop, so enforce that some maximum is specified.
        if not allow_inf_max_len:
          assert res is not True, "%r: specify max_seq_len" % rec_layer
        if seq_len_info is not None:
          end_flag, _ = seq_len_info
          any_not_ended = tf.reduce_any(tf.logical_not(end_flag), name="any_not_ended")
          res = opt_logical_and(res, any_not_ended)
        return res

    from returnn.tf.util.basic import constant_with_shape
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
      end_layer_batch_dim = self.layer_data_templates["end"].get_batch_dim()
      init_seq_len_info = (
        constant_with_shape(False, shape=[end_layer_batch_dim], name="initial_end_flag"),
        constant_with_shape(0, shape=[end_layer_batch_dim], name="initial_seq_len"))
      init_loop_vars += (init_seq_len_info,)
      shape_invariants += ((tf.TensorShape([None]), tf.TensorShape([None])),)
    if self.layers_in_loop:
      final_loop_vars = self._while_loop(
        cond=cond,
        body=body,
        loop_vars=init_loop_vars,
        shape_invariants=shape_invariants)
      if have_known_seq_len:
        _, final_net_vars, final_acc_tas = final_loop_vars
        assert fixed_seq_len is not None
        seq_len = fixed_seq_len
        if output_beam:
          assert not input_beam or input_beam == output_beam, (
            "%s: input beam %r, output beam %r, sources %r, target %r" % (
              self.parent_rec_layer, input_beam, output_beam,
              self.parent_rec_layer.sources, self.parent_rec_layer.target))
          assert output_template.output.batch.beam == output_beam
          time_dim_tag = time_dim_tag.get_for_batch_ctx(
            batch=output_template.output.batch, ctx=self.net.control_flow_ctx)
          assert time_dim_tag.dyn_size is not None
          seq_len = time_dim_tag.dyn_size
      else:
        _, final_net_vars, final_acc_tas, (_, seq_len) = final_loop_vars
        # Note: In case of search, the seq len would have the beam from the end layer.
        # This might not be the same as the final output beam.
        # This should correctly be resolved in _construct_output_layers_moved_out and _opt_search_resolve.
        # So we do not assign it to the dim tag at this point.
        assert "output" in extra_output_layers
        max_seq_len = tf.reduce_max(seq_len, name="dyn_max_seq_len")
      self.get_final_rec_vars = lambda layer_name_: self.get_layer_rec_var_from_loop_vars(
        loop_vars=final_net_vars, layer_name=layer_name_, final_frame=True, seq_len=seq_len)
      assert isinstance(final_acc_tas, list)
      if len(outputs_to_accumulate) > 0:
        assert isinstance(final_acc_tas[0], tf.TensorArray)
      assert len(final_acc_tas) == len(outputs_to_accumulate)
      self.final_acc_tas_dict = {
        out.name: out.get_final_tensor_array(final_acc_ta)
        for (final_acc_ta, out) in zip(final_acc_tas, outputs_to_accumulate)}  # type: typing.Dict[str,typing.Optional[tf.TensorArray]]  # nopep8
      self._assign_last_frames_from_loop_vars(final_net_vars)
    else:  # no layers inside loop, all optimized out
      seq_len = None
      final_net_vars = None
      self.get_final_rec_vars = None
      self.final_acc_tas_dict = None

    self._construct_output_layers_moved_out(
      loop_accumulated=self.final_acc_tas_dict, seq_len=seq_len,
      extra_output_layers=extra_output_layers, final_net_vars=final_net_vars)
    self._assign_last_frames_from_optimized()

    if layer_names_with_losses:
      from returnn.tf.network import LossHolder
      with tf.name_scope("sub_net_loss"):
        # Losses from layers moved out of the loop.
        for layer_name in sorted(layer_names_with_losses):
          if layer_name in self.input_layers_moved_out + self.output_layers_moved_out:
            if layer_name in self.input_layers_moved_out:
              layer_with_loss_inst = self.input_layers_net.layers[layer_name]
            else:
              layer_with_loss_inst = self.output_layers_net.layers[layer_name]
            assert isinstance(layer_with_loss_inst, LayerBase)
            for loss in layer_with_loss_inst.get_losses_initialized():
              assert loss.name not in self.accumulated_losses, "loss name not unique"
              self.accumulated_losses[loss.name] = loss

        if accumulated_loop_losses:
          # Now collect the losses from layers inside the loop.
          with tf.name_scope("sub_loss_normalization_factor"):
            sub_loss_normalization_factor = 1.0 / tf.cast(tf.reduce_sum(seq_len), tf.float32)
          for _, loss in sorted(accumulated_loop_losses.items()):
            assert isinstance(loss, LossHolder)
            assert loss.loss.layer, "sub loss init not called?"
            assert loss.name not in self.accumulated_losses, "loss name not unique"
            loss_value = tensor_array_stack(
              self.final_acc_tas_dict["loss_%s" % loss.name], stop=max_seq_len,
              name="loss_%s_stack" % get_valid_scope_name_from_str(loss.name))
            if self.final_acc_tas_dict["error_%s" % loss.name] is not None:
              error_value = tensor_array_stack(
                self.final_acc_tas_dict["error_%s" % loss.name], stop=max_seq_len,
                name="error_%s_stack" % get_valid_scope_name_from_str(loss.name))
            else:
              error_value = None
            loss_value.set_shape(tf.TensorShape((None, None)))  # (time, batch)
            if error_value is not None:
              error_value.set_shape(tf.TensorShape((None, None)))  # (time, batch)
            loss_wrapped = _SubnetworkRecWrappedLoss(
              base_loss=loss.loss,
              loss_value=loss_value, error_value=error_value,
              norm_factor=sub_loss_normalization_factor,
              seq_lens=seq_len)
            self.accumulated_losses[loss.name] = LossHolder(
              name=loss.name,
              layer=loss.loss.layer,
              layer_output=rec_layer.output,  # not the correct output, but we only use it to check e.g. for time-dim
              loss=loss_wrapped)

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

    for key in (
          self.net.used_data_keys |
          (self.input_layers_net.used_data_keys if self.input_layers_net else set()) |
          (self.output_layers_net.used_data_keys if self.output_layers_net else set())):
      if key == "source":
        continue
      self.parent_net.used_data_keys.add(key)

    with tf.name_scope("output"):
      output_layer = None
      if self.input_layers_net and "output" in self.input_layers_net.layers:
        output_layer = self.input_layers_net.layers["output"]
      elif self.output_layers_net and "output" in self.output_layers_net.layers:
        output_layer = self.output_layers_net.layers["output"]
      if output_layer:
        assert isinstance(output_layer, LayerBase)
        output_data = output_layer.output.copy_as_time_major()
        if not self.time_dim_tag.is_dim_known():
          self.time_dim_tag.declare_same_as(output_data.get_time_dim_tag())
        elif self.time_dim_tag not in output_data.dim_tags:
          # We allow this for older behavior version to not break some older setups.
          # https://github.com/rwth-i6/returnn/issues/1140
          BehaviorVersion.require(
            False,
            "%s: time-dim-tag mismatch: self %r vs sub-output-layer %r time-dim-tag %r" % (
              rec_layer, self.time_dim_tag, output_data, output_data.get_time_dim_tag()), version=13)
          # No further checks, it would fail anyway.
          # Replace the actual rec layer output and return.
          rec_layer.output = output_data
          return output_data.placeholder
        assert len(rec_layer.output.dim_tags) == len(output_data.dim_tags)
        for tag1, tag2 in zip(rec_layer.output.dim_tags, output_data.dim_tags):
          try:
            assert tag1.is_equal(tag2, allow_same_feature_dim=True), (
              "%s vs %s from %s" % (rec_layer.output, output_data, output_layer))
          except Exception as exc:
            self._handle_construct_exception(description="output format match", exception=exc)
            raise
          # Make sure they are the same.
          # It can happen that they are not when the dim tag is created inside,
          # and then created once for the template layer, and again for the real layer.
          # Make sure they are really the same such that we get all information like dyn sizes.
          tag1.declare_same_as(tag2)
        return output_data.placeholder
      else:
        assert seq_len is not None
        rec_layer.output.size_placeholder[0] = seq_len
        assert not self.net.layers["output"].get_search_choices()
        return tensor_array_stack(
          self.final_acc_tas_dict["output_output"], stop=max_seq_len, name="output_stack")  # e.g. (time, batch, dim)

  def _get_search_choice_seq(self, search_choices):
    """
    :param SearchChoices search_choices:
    :rtype: list[LayerBase]
    """
    layer_choice = search_choices.owner
    assert not layer_choice.name.startswith("prev:")
    assert not isinstance(layer_choice, _TemplateLayer)

    # There can be multiple choices in a single rec step. Collect them.
    # Find next choice layer. Then iterate through its source choice layers through time
    # and resolve the output over time to be in line with the final output search choices.

    def get_choice_seq(choice_base):
      """
      :param LayerBase choice_base:
      :return: choice_seq, prev_choice
      :rtype: (list[LayerBase], _TemplateLayer)
      """
      choice_seq = [choice_base]
      choice = choice_base
      while True:
        assert choice.network is self.net, "not yet implemented otherwise"
        assert ("choice_%s" % choice.name) in self.final_acc_tas_dict
        assert choice.search_choices
        assert choice.search_choices.src_layer
        choice = choice.search_choices.src_layer
        if isinstance(choice, _TemplateLayer):
          assert choice.is_prev_time_frame
          return choice_seq, choice
        choice_seq.append(choice)

    # Get the very latest choice in the rec layer.
    latest_layer_choice = layer_choice
    _, latest_layer_choice = get_choice_seq(latest_layer_choice)
    assert latest_layer_choice.name.startswith("prev:")
    latest_layer_choice = self.net.layers[latest_layer_choice.name[len("prev:"):]]
    assert latest_layer_choice.search_choices

    # Get the whole choice sequence, starting from the latest to the first.
    choice_seq_in_frame, prev_frame_choice = get_choice_seq(latest_layer_choice)
    assert prev_frame_choice.name == "prev:%s" % latest_layer_choice.name
    assert layer_choice in choice_seq_in_frame
    assert choice_seq_in_frame[0] is latest_layer_choice
    return choice_seq_in_frame

  def _opt_search_resolve(self, layer_name, acc_ta, final_net_vars, seq_len, search_choices_cache):
    """
    This assumes that we have frame-wise accumulated outputs of the specific layer (acc_ta).
    If that layer depends on frame-wise search choices, i.e. if the batch dim includes a search beam,
    and each frame has different beams,
    then we will resolve that to the common final best search choices,
    which is determined by the latest search choice given by the layer.
    This assumes that we also have `self.final_acc_tas_dict["choice_%s" % choice_base.name]` available.
    In addition, we resolve the sequence lengths (whose beams correspond to the end layer) to the final search choices.

    :param str layer_name:
    :param tf.TensorArray acc_ta: accumulated outputs of that layer
    :param final_net_vars:
    :param tf.Tensor seq_len: shape (batch * beam,), has beam of the "end" layer in case of dynamic sequence lengths,
      otherwise beam of rec_layer.output
    :param dict[str,SearchChoices] search_choices_cache: inner search choices layer -> final search choices
    :return: (new acc_ta, latest layer choice name, search choices, resolved seq_len)
    :rtype: (tf.TensorArray,str|None,SearchChoices|None,tf.Tensor)
    """
    import os
    from returnn.tf.util.basic import nd_indices, assert_min_tf_version, expand_dims_unbroadcast
    from returnn.tf.util.basic import get_shape_dim, get_valid_scope_name_from_str
    from returnn.tf.util.basic import TensorCachedComputation
    rec_layer = self.parent_rec_layer
    try:
      layer = self.net.get_layer(layer_name)
    except LayerNotFound:  # layer is not inside loop
      return acc_ta, None, None, seq_len
    search_choices = layer.get_search_choices()
    if not search_choices:
      return acc_ta, None, None, seq_len
    if search_choices.owner.network is not self.net:
      return acc_ta, None, search_choices, seq_len
    if search_choices.keep_raw:
      search_choices_cache[search_choices.owner.name] = search_choices
      # Make a new seq_len tensor, to be able to attach a new dim tag to it.
      # This is needed as long as we make use of get_tag_from_size_tensor.
      # Cache it such that we have it unique.
      cache = TensorCachedComputation(search_choices, key=("seq_len_raw", seq_len))
      if not cache.has_cache():
        assert search_choices.owner.output.batch and search_choices.owner.output.batch.beam
        seq_len = tf.identity(seq_len)
        seq_len._RETURNN_dyn_size_beam = search_choices.owner.output.batch.beam
        cache.set_cache(seq_len)
      seq_len = cache.get_cache()
      return acc_ta, search_choices.owner.name, search_choices, seq_len
    layer_choice = search_choices.owner
    is_prev_choice = False
    if isinstance(layer_choice, _TemplateLayer):
      assert layer_choice.is_prev_time_frame
      assert layer_choice.name.startswith("prev:")
      layer_choice = self.net.layers[layer_choice.name[len("prev:"):]]
      assert layer_choice.search_choices
      search_choices = layer_choice.search_choices
      is_prev_choice = True

    choice_seq_in_frame = self._get_search_choice_seq(search_choices)
    latest_layer_choice = choice_seq_in_frame[0]

    # The max_seq_len might actually be one more, as it includes the EOS, but that does not matter;
    # we just want to create a new acc_ta with the same length.
    # (Cutting off the EOS is handled elsewhere.)
    max_seq_len = acc_ta.size()
    initial_i = tf.identity(max_seq_len - 1, name="search_resolve_initial_i")  # we go backwards
    latest_beam_size = latest_layer_choice.output.beam.beam_size
    batch_dim = rec_layer.network.get_data_batch_dim()

    initial_beam_choices = tf.range(0, latest_beam_size)  # (beam_out,)
    initial_beam_choices = expand_dims_unbroadcast(
      initial_beam_choices, axis=0, dim=batch_dim)  # (batch, beam_out)

    # Translate the beams of seq_lens to correspond to the latest choice
    try:
      end_layer = self.net.get_layer("end")
    except LayerNotFound:
      # This means have_known_seq_len=True.
      end_layer = None

    if end_layer:
      # seq_len is determined from the end-layer. We need to translate it to the right beam.
      end_layer_choice = self.net.get_search_choices(src=end_layer)
      assert end_layer_choice and end_layer_choice.search_choices
      if end_layer_choice.name.startswith("prev:"):
        # Logic from maybe_transform. It would be translated to the current beam.
        end_layer_choice = self.net.layers[end_layer_choice.name[len("prev:"):]]
      assert end_layer_choice in choice_seq_in_frame, (
        "End layer must not have a beam independent from output layer '{}'.".format(layer_name))

      end_layer_choice_index = choice_seq_in_frame.index(end_layer_choice)
      choices_seq_until_end_layer = choice_seq_in_frame[:end_layer_choice_index]

      for choice_ in reversed(choices_seq_until_end_layer):
        src_choice_beams = self.final_acc_tas_dict["choice_%s" % choice_.name].read(
          max_seq_len - 1, name="ta_read_choice")  # (batch, beam) -> beam_in idx
        seq_len = tf_util.select_src_beams(seq_len, src_choice_beams)
        assert choice_.output.batch and choice_.output.batch.beam
        assert getattr(seq_len, "_RETURNN_dyn_size_beam", NotSpecified) in (NotSpecified, choice_.output.batch.beam)
        seq_len._RETURNN_dyn_size_beam = choice_.output.batch.beam

    else:  # not end_layer
      # Here we don't need to resolve anything, as the sequence length is the same for all hyps in the beam.
      # However, beam size for the current output may be different from the "output" layer.
      tag = Dim.get_tag_from_size_tensor(seq_len)
      assert tag
      latest_batch = (
        latest_layer_choice.output.batch
        or self.parent_rec_layer.output.batch.copy_set_beam(latest_layer_choice.output.beam))
      tag = tag.get_for_batch_ctx(batch=latest_batch, ctx=self.net.control_flow_ctx)
      assert tag.dyn_size is not None
      assert tag.batch == latest_batch and tag.batch.beam == latest_layer_choice.output.beam
      seq_len = tag.dyn_size

    new_acc_output_ta = tf.TensorArray(
      name="search_resolved_%s" % os.path.basename(
        (acc_ta.handle if acc_ta.handle is not None else acc_ta.flow).op.name),
      dtype=layer.output.dtype,
      element_shape=tf.TensorShape(layer.output.batch_shape),
      size=max_seq_len,
      infer_shape=True)

    def transform(i, idxs_exp, new_acc_output_ta_):
      """
      :param int|tf.Tensor i: scalar, int32
      :param tf.Tensor idxs_exp: (batch, beam_out, 2) -> (batch idx, beam idx)
      :param tf.TensorArray new_acc_output_ta_:
      :return: new_acc_output_ta_
      :rtype: tf.TensorArray
      """
      with tf.name_scope("transform_output_%s" % get_valid_scope_name_from_str(layer_name)):
        output_ = acc_ta.read(i)  # (batch * beam, [n_out])
        assert layer.output.batch_dim_axis == 0
        out_shape = list(layer.output.batch_shape)
        output_.set_shape(tf.TensorShape(out_shape))
        for i_, d in list(enumerate(out_shape)):
          if i_ == 0:
            continue  # not relevant
          if d is None:
            out_shape[i_] = tf.shape(output_)[i_]
        output_ = tf.reshape(
          output_, [batch_dim, layer_choice.output.beam.beam_size] + out_shape[1:],
          name="split_batch_beam")  # (batch, beam, [n_out])
        output_ = tf.gather_nd(output_, idxs_exp)  # (batch, beam_par, [n_out])
        beam_par = get_shape_dim(idxs_exp, 1)
        output_ = tf.reshape(
          output_, [batch_dim * beam_par] + out_shape[1:],
          name="merge_batch_beam")  # (batch * beam_par, [n_out])
        return new_acc_output_ta_.write(i, output_)

    def search_resolve_body(i, choice_beams, new_acc_output_ta_):
      """
      This loops goes backwards through time.
      Similar to tf.contrib.seq2seq.GatherTree.

      :param tf.Tensor i: starts at max_seq_len - 1
      :param tf.Tensor choice_beams: from the previous step, shape (batch, beam_out) -> beam idx,
        just before prev_frame_choice
      :param tf.TensorArray|None new_acc_output_ta_: shape (batch * beam, n_out) per frame
      :return: (i, choice_beams, new_acc_output_ta)
      :rtype: (tf.Tensor, tf.Tensor, tf.TensorArray|None)
      """
      # noinspection PyProtectedMember
      with reuse_name_scope((rec_layer._rec_scope.name or "rec") + "/while_loop_search_resolve_body", absolute=True):
        # We start at the output layer choice base, and search for its source, i.e. for the previous time frame.
        # noinspection PyShadowingNames
        for choice_ in choice_seq_in_frame:
          with tf.name_scope("choice_beams_%s" % get_valid_scope_name_from_str(choice_.name)):
            assert_min_tf_version((1, 1), "gather_nd")
            idxs_exp = nd_indices(choice_beams)  # (batch, beam_out, 2) -> (batch idx, beam idx)
            # noinspection PyShadowingNames
            src_choice_beams = self.final_acc_tas_dict["choice_%s" % choice_.name].read(
              i, name="ta_read_choice")  # (batch, beam) -> beam_in idx
            assert src_choice_beams.get_shape().ndims == 2
            # noinspection PyShadowingNames
            src_choice_beams = tf.gather_nd(src_choice_beams, idxs_exp)  # (batch, beam_out)

          if new_acc_output_ta_ is not None and choice_ == layer_choice:
            new_acc_output_ta_ = transform(
              (i + 1) if is_prev_choice else i, idxs_exp, new_acc_output_ta_)

          choice_beams = src_choice_beams

        return (
          i - 1,
          src_choice_beams,
          new_acc_output_ta_)

    # noinspection PyUnusedLocal
    def search_resolve_cond(i, *args):
      """
      :param tf.Tensor i: rec step index, scalar, int32
      :return: whether to continue with this i, scalar, bool
      :rtype: tf.Tensor
      """
      return tf.greater_equal(i, 0, name="search_resolve_loop_cond_i_ge_0")

    if is_prev_choice:
      # Resolve first the choices from last frame.
      with tf.name_scope("search_resolve_last_frame"):
        initial_i, initial_beam_choices = tf.cond(
          search_resolve_cond(initial_i),
          lambda: search_resolve_body(initial_i, initial_beam_choices, None)[:2],
          lambda: (initial_i, initial_beam_choices))

    final_i, final_beam_choices, new_acc_output_ta = tf.while_loop(
      name="search_resolve_loop",
      cond=search_resolve_cond,
      body=search_resolve_body,
      loop_vars=(initial_i, initial_beam_choices, new_acc_output_ta),
      back_prop=self.parent_rec_layer.back_prop)

    if is_prev_choice:
      # Final missing first frame.
      beam_choices = tf.zeros_like(final_beam_choices)
      with tf.name_scope("search_resolve_first_frame"):
        new_acc_output_ta = tf.cond(
          tf.less(0, max_seq_len),
          lambda: transform(0, nd_indices(beam_choices), new_acc_output_ta),
          lambda: new_acc_output_ta)

    # Create the search choices for the rec layer accumulated output itself.
    # The beam scores will be of shape (batch, beam).
    if latest_layer_choice.name not in search_choices_cache:
      acc_search_choices = SearchChoices(owner=latest_layer_choice, beam_size=latest_beam_size)
      final_choice_rec_vars = self.get_layer_rec_var_from_loop_vars(
        loop_vars=final_net_vars,
        layer_name=latest_layer_choice.name,
        final_frame=True,
        seq_len=seq_len)
      acc_search_choices.set_beam_from_rec(final_choice_rec_vars)
      search_choices_cache[latest_layer_choice.name] = acc_search_choices

    return new_acc_output_ta, latest_layer_choice.name, search_choices, seq_len

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
    # Note, that layers_in_loop will also contain sublayers as separate entries (added via layer.dependencies)
    # because we might need to accumulate their outputs into separate TensorArrays.
    layers_in_loop = []  # type: typing.List[_TemplateLayer]

    def visit(deps):
      """
      :param list[LayerBase] deps:
      """
      for layer in deps:
        if not isinstance(layer, _TemplateLayer):  # real layer from base net or so
          continue
        if layer.name == "data" or layer.name.startswith("data:"):
          continue
        assert self.layer_data_templates[layer.name] is layer
        if layer not in layers_in_loop:
          layers_in_loop.append(layer)
          visit(layer.dependencies)
    visit([self.layer_data_templates[name] for name in needed_outputs])

    self.input_layers_moved_out = []  # type: typing.List[str]
    self.output_layers_moved_out = []  # type: typing.List[str]

    def output_can_move_out(layer, _collocated=None):
      """
      :param _TemplateLayer layer:
      :param set[_TemplateLayer]|None _collocated:
      :rtype: bool
      """
      if _collocated is None:
        _collocated = {layer}
      else:
        _collocated.add(layer)
      assert isinstance(layer, _TemplateLayer)
      # Special case: end-layer, which is added if the seq-len is unknown, cannot be moved out.
      if layer.name == "end":
        return False
      if layer.search_choices:
        return False  # need to perform the search inside the loop currently
      if layer in layer.dependencies:  # recursive on itself
        return False
      # layer.output is used by other layers?
      for other_layer in layers_in_loop:
        if other_layer in _collocated:
          continue
        if other_layer.name in layer.collocate_with or layer.name in other_layer.collocate_with:
          if output_can_move_out(other_layer, _collocated=_collocated):
            continue  # this is ok, we can move all out
          else:
            return False
        if layer in other_layer.dependencies:
          return False
      return True

    def find_output_layer_to_move_out():
      """
      :rtype: _TemplateLayer|None
      """
      for layer in layers_in_loop:
        if output_can_move_out(layer):
          return layer
      return None

    def output_move_out(layer):
      """
      :param _TemplateLayer layer:
      """
      assert isinstance(layer, _TemplateLayer)
      layers_in_loop.remove(layer)
      self.output_layers_moved_out.append(layer.name)

    def input_can_move_out(layer):
      """
      :param _TemplateLayer layer:
      :rtype: bool
      """
      assert isinstance(layer, _TemplateLayer)
      if layer.name in [":i", "end"]:  # currently not fully implemented
        return False
      if layer.search_choices:
        return False  # need to perform the search inside the loop currently
      layer_deps = layer.dependencies
      # We depend on other layers from this sub-network?
      for other_layer in layers_in_loop:
        if other_layer in layer_deps:
          return False
        if other_layer.name in layer.collocate_with:
          return False
      return True

    def find_input_layer_to_move_out():
      """
      :rtype: _TemplateLayer|None
      """
      for layer in layers_in_loop:
        if input_can_move_out(layer):
          return layer
      return None

    def input_move_out(layer):
      """
      :param _TemplateLayer layer:
      """
      assert isinstance(layer, _TemplateLayer)
      layers_in_loop.remove(layer)
      self.input_layers_moved_out.append(layer.name)

    # First try out to move as much output-layers as possible.
    while True:
      output_layer = find_output_layer_to_move_out()
      if output_layer:
        output_move_out(output_layer)
      else:
        break
    # Now, both input-layers and output-layers.
    while True:
      output_layer = find_output_layer_to_move_out()
      if output_layer:
        output_move_out(output_layer)
        continue  # greedy on output layers
      input_layer = find_input_layer_to_move_out()
      if input_layer:
        input_move_out(input_layer)
      if not output_layer and not input_layer:
        break

    self.layers_in_loop = [layer.name for layer in layers_in_loop]

    log_stream = log.v3
    print("Rec layer %r (search %s, train %s) sub net:" % (
        self.parent_rec_layer.get_absolute_name(), self.net.search_flag,
        repr(self.net.train_flag.name) if isinstance(self.net.train_flag, tf.Tensor) else self.net.train_flag),
      file=log_stream)
    remaining_layers = set(self.net_dict.keys())

    def dump_info(s, ls):
      """
      :param str s:
      :param list[str] ls:
      """
      print("  %s: (#: %i)" % (s, len(ls)), file=log_stream)
      for layer_name in ls:
        print("    %s" % layer_name, file=log_stream)
        if layer_name in remaining_layers:  # sub-layers are not in the net_dict, or auto-constructed like ":i"
          remaining_layers.remove(layer_name)
      if not ls:
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

    from returnn.tf.network import TFNetwork, ExternData
    from .base import InternalLayer
    from returnn.tf.util.basic import concat_with_opt_broadcast
    self.input_layers_net = TFNetwork(
      name="%s/%s(rec-subnet-input)" % (
        self.parent_net.name, self.parent_rec_layer.name if self.parent_rec_layer else "?"),
      extern_data=ExternData(),
      train_flag=self.parent_net.train_flag,
      search_flag=self.parent_net.search_flag,
      over_rec_time_dim=self.time_dim_tag,
      over_rec_time_dim_subs=self._time_dim_tags,
      parent_layer=self.parent_rec_layer,
      parent_net=self.parent_net)
    self.input_layers_net.is_root_in_ctx = True
    self.input_layers_net.layers_desc.update(self.net_dict)
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
        output.placeholder = concat_with_opt_broadcast(
          [initial_wt, output.placeholder], allow_broadcast=[True, False], axis=0, name="concat_in_time")
        output.placeholder = output.placeholder[:-1]  # remove last frame
        # Note: This seq_len might make sense to use here:
        # output.size_placeholder[0] = tf.minimum(output.size_placeholder[0] + 1, tf.shape(x)[0])
        # However, often we assume that we keep the same seq lens as the output layer.
        assert isinstance(self.input_layers_net, TFNetwork)
        layer = self.input_layers_net.add_layer(
          name="prev:%s" % name, output=output, layer_class=InternalLayer, sources=[cur_layer])
        return layer

    # get_layer similar to in self._construct() but simplified.
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
        return self._get_parent_layer(name[len("base:"):])
      if name not in self.layer_data_templates:
        raise LayerNotFound(
          "layer not found: %s/%s" % (self.input_layers_net, name),
          layer_name=name, network=self.input_layers_net)
      # noinspection PyBroadException
      try:
        return self.input_layers_net.construct_layer(self.net_dict, name=name, get_layer=get_layer)
      except Exception as exc:
        self._handle_construct_exception(description="input-net construction of layer %r" % name, exception=exc)
        raise

    # Same scope as the main subnet, so that it stays compatible.
    # noinspection PyProtectedMember
    with reuse_name_scope(self.parent_rec_layer._rec_scope):
      for layer_name in self.input_layers_moved_out:
        get_layer(layer_name)

    self._update_time_dim_tags_potential_set_new_hashes()

  def _construct_output_layers_moved_out(self, loop_accumulated, seq_len, extra_output_layers, final_net_vars):
    """
    See self._move_outside_loop().
    The output layers will be constructed in self.output_layers_net.

    :param dict[str,tf.TensorArray]|None loop_accumulated:
      keys, see self.get_output(). should be like "output_<layer_name>"
    :param tf.Tensor|None seq_len: shape (batch,). None if no loop_accumulated
    :param set[str] extra_output_layers:
    :param final_net_vars:
    :return: nothing, will init self.output_layers_net
    """
    if not self.output_layers_moved_out and not extra_output_layers:
      return
    from returnn.tf.util.basic import tensor_array_stack, concat_with_opt_broadcast
    from returnn.tf.network import TFNetwork, ExternData
    from .base import InternalLayer, WrappedInternalLayer
    from .basic import LengthLayer

    self.output_layers_net = TFNetwork(
      name="%s/%s(rec-subnet-output)" % (
        self.parent_net.name, self.parent_rec_layer.name if self.parent_rec_layer else "?"),
      extern_data=ExternData(),
      train_flag=self.parent_net.train_flag,
      search_flag=self.parent_net.search_flag,
      over_rec_time_dim=self.time_dim_tag,
      over_rec_time_dim_subs=self._time_dim_tags,
      parent_layer=self.parent_rec_layer,
      parent_net=self.parent_net)
    self.output_layers_net.is_root_in_ctx = True
    self.output_layers_net.layers_desc.update(self.net_dict)
    if self.parent_rec_layer.input_data:
      self.output_layers_net.extern_data.data["source"] = \
        self.parent_rec_layer.input_data
    for key in self.parent_net.extern_data.data.keys():
      self.output_layers_net.extern_data.data[key] = \
        self.parent_net.extern_data.data[key]

    prev_layers = {}  # type: typing.Dict[str,WrappedInternalLayer]
    loop_acc_layers = {}  # type: typing.Dict[str,WrappedInternalLayer]
    search_choices_cache = {}  # type: typing.Dict[str,SearchChoices]  # inner layer -> acc search choices
    loop_acc_layers_search_choices = {}  # type: typing.Dict[str,str]  # loop acc layer -> inner layer

    # noinspection PyShadowingNames
    def get_loop_acc_layer(name):
      """
      :param str name:
      :rtype: LayerBase
      """
      assert loop_accumulated is not None, "no layers in loop"
      if name in loop_acc_layers:
        return loop_acc_layers[name]
      in_loop_layer = self.net.get_layer(name)
      with tf.name_scope(in_loop_layer.tf_scope_name):
        acc_ta = loop_accumulated["output_%s" % name]
        acc_ta, latest_layer_choice_name, search_choices, resolved_seq_len = self._opt_search_resolve(
          layer_name=name, acc_ta=acc_ta, final_net_vars=final_net_vars, seq_len=seq_len,
          search_choices_cache=search_choices_cache)
        # Use the output Data from the in-loop layer,
        # as this might have set dyn sizes on dim tags.
        is_out_time_dim = search_choices == self.layer_data_templates["output"].get_search_choices()
        time_dim_tag = Dim.get_tag_from_size_tensor(resolved_seq_len)
        if not time_dim_tag:
          if is_out_time_dim:
            time_dim_tag = self.time_dim_tag
          else:
            time_dim_tag = Dim(
              kind=Dim.Types.Spatial,
              description="dyn-time:%s/%s" % (self.parent_rec_layer.get_full_ctx_name(), search_choices),
              auto_generated=True)
        elif is_out_time_dim:
          self.time_dim_tag.declare_same_as(time_dim_tag)
        output = (
          in_loop_layer.output
          .copy_template()
          .copy_add_dim_by_tag(time_dim_tag, unbroadcast=True, axis=0)
          .copy_template_set_ctx(self.parent_net.get_control_flow_ctx()))
        if latest_layer_choice_name:
          output.beam = self.net.layers[latest_layer_choice_name].search_choices.get_beam_info()
        elif search_choices:
          output.beam = search_choices.get_beam_info()
        else:
          output.beam = None
        if output.batch:
          output.batch = output.batch.copy_set_beam(output.beam)
        output.set_dynamic_size(axis=0, sizes=resolved_seq_len)
        max_len = tf.reduce_max(resolved_seq_len)
        # We should have accumulated it.
        output.placeholder = tensor_array_stack(acc_ta, stop=max_len)  # e.g. (time,batch,dim)
        assert isinstance(self.output_layers_net, TFNetwork)
        layer_ = self.output_layers_net.add_layer(
          name=name, output=output, layer_class=WrappedInternalLayer, sources=[], base_layer=in_loop_layer)
        if latest_layer_choice_name:
          loop_acc_layers_search_choices[name] = latest_layer_choice_name
        loop_acc_layers[name] = layer_
        if isinstance(in_loop_layer, LengthLayer):
          tag = in_loop_layer.dim_tag.get_for_batch_ctx(layer_.output.batch, layer_.output.control_flow_ctx)
          if not tag.dyn_size_ext:
            tag.dyn_size_ext = layer_.output
        return layer_

    # noinspection PyShadowingNames
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
        output.placeholder = concat_with_opt_broadcast(
          [initial_wt, output.placeholder], allow_broadcast=[True, False], axis=0, name="concat_in_time")
        output.placeholder = output.placeholder[:-1]  # remove last frame
        # Note: This seq_len might make sense to use here:
        # output.size_placeholder[0] = tf.minimum(output.size_placeholder[0] + 1, tf.shape(x)[0])
        # However, often we assume that we keep the same seq lens as the output layer.
        # output.size_placeholder[0] = seq_len. just don't modify. assert seq_len is not None
        assert isinstance(self.output_layers_net, TFNetwork)
        layer = self.output_layers_net.add_layer(
          name="prev:%s" % name, output=output, layer_class=WrappedInternalLayer, base_layer=cur_layer)
        prev_layers[name] = layer
        return layer

    # get_layer similar to in self._construct() but simplified.
    # noinspection PyShadowingNames
    def get_layer(name):
      """
      :param str name:
      :rtype: LayerBase
      """
      if name.startswith("prev:"):
        return get_prev_layer(name[len("prev:"):])
      if name.startswith("base:"):
        return self._get_parent_layer(name[len("base:"):])
      if name in self.input_layers_moved_out:
        return self.input_layers_net.get_layer(name)
      if name in self.output_layers_moved_out or name.startswith("data:") or name == "data":
        # noinspection PyBroadException
        try:
          return self.output_layers_net.construct_layer(self.net_dict, name=name, get_layer=get_layer)
        except Exception as exc:
          self._handle_construct_exception(description="output-net construction of layer %r" % name, exception=exc)
          raise
      if name not in self.layer_data_templates:
        raise LayerNotFound(
          "layer not found: %s/%s" % (self.output_layers_net, name),
          layer_name=name, network=self.output_layers_net)
      # It means that the layer is inside the loop.
      return get_loop_acc_layer(name)

    # Same scope as the main subnet, so that it stays compatible.
    # noinspection PyProtectedMember
    with reuse_name_scope(self.parent_rec_layer._rec_scope):
      for layer_name in sorted(extra_output_layers):
        self.output_layers_net.layers[layer_name] = get_layer(layer_name)
      for layer_name in self.output_layers_moved_out:
        get_layer(layer_name)

    # We want to have one single layer with search choices.
    for name, search_choices in search_choices_cache.items():
      if name not in self.output_layers_net.layers:
        # Create dummy layer.
        output = (
          self.layer_data_templates[name].output
          .copy_template_adding_time_dim(time_dim_axis=0)
          .copy_template_set_ctx(self.output_layers_net.get_control_flow_ctx()))
        output.beam = search_choices.get_beam_info()
        layer = InternalLayer(name=name, network=self.output_layers_net, output=output)
        self.output_layers_net.layers[name] = layer
      else:
        layer = self.output_layers_net.layers[name]
      # Set the search choices only for this layer.
      layer.search_choices = search_choices
      search_choices.owner = layer
      # Now mark other layers with the same search choices dependent on this layer.
      for name_, name__ in loop_acc_layers_search_choices.items():
        if name__ == name:
          layer_ = self.output_layers_net.layers[name_]
          assert isinstance(layer_, InternalLayer)
          layer_.sources.append(layer)

    # Now, after constructing all, maybe reset the time-dim-axis.
    # It is valid during the construction that layers set any time-dim-axis they want,
    # and this can be even mandatory, such that layers like SoftmaxOverSpatialLayer act as requested.
    # However, after construction, when accessing any of these layers,
    # we would expect that their time-dim-axis matches the same as from the rec loop.
    for layer in self.output_layers_net.layers.values():
      layer.output.mark_same_time(self._time_dim_tags)

    self._update_time_dim_tags_potential_set_new_hashes()

  def _update_time_dim_tags_potential_set_new_hashes(self):
    """
    This is quite counter-intuitive and potential dangerous:
    self._time_dim_tags is a set over dim tags.
    A set uses the hash values of its objects.
    Via Dim.declare_same_as,
    we can change the hash value of dim tags.
    The set would not automatically be updated though,
    and then you can end up with the strange case::

      (list(self._time_dim_tags) == [self.time_dim_tag]
       and self.time_dim_tag not in self._time_dim_tags)

    Additionally, e.g. the input layer net might have references to the set,
    so we cannot simply create a new set.
    """
    # Clear the set and re-add all tags. That will use any potential updated hash values.
    tags = list(self._time_dim_tags)
    self._time_dim_tags.clear()
    self._time_dim_tags.update(tags)


RecLayer.SubnetworkRecCell = _SubnetworkRecCell


class _TemplateLayer(LayerBase):
  """
  Used by _SubnetworkRecCell.
  In a first pass, it creates template layers with only the meta information about the Data.
  All "prev:" layers also stay instances of _TemplateLayer in the real computation graph.
  """

  def __init__(self, network, name, construct_stack=None, cell=None):
    """
    :param returnn.tf.network.TFNetwork network:
    :param str name:
    :param LayerBase|None construct_stack: just for debugging repr
    :param _SubnetworkRecCell|None cell:
    """
    # Init with some dummy.
    dummy_data = Data(
      name="%s:dummy_initial_template_data" % name,
      dim_tags=[],
      control_flow_ctx=network.get_control_flow_ctx())
    super(_TemplateLayer, self).__init__(output=dummy_data, name=name, network=network)
    self.output.size_placeholder = {}  # must be initialized
    self.layer_class = ":uninitialized-template"
    self.is_data_template = False
    self.is_prev_time_frame = False
    self.is_initialized = False
    self.layer_class_type = None  # type: typing.Optional[typing.Type[LayerBase]]
    self.kwargs = None  # type: typing.Optional[typing.Dict[str]]  # after transform_config_dict
    self.dependencies = []  # type: typing.List[LayerBase]
    self.cur_frame_dependencies = []  # type: typing.List[LayerBase]
    self.prev_frame_dependencies = []  # type: typing.List[_TemplateLayer]
    self.used_by = []  # type: typing.List[LayerBase]  # reverse deps
    self.construct_stack = construct_stack
    self._template_base = None  # type: typing.Optional[_TemplateLayer]
    self._cell = cell

  def __repr__(self):
    if self.is_initialized:
      return "<%s(%s)(%s) %s%r out_type=%s (construction stack %r)>" % (
        self.__class__.__name__, self.layer_class_type.__name__ if self.layer_class_type else None, self.layer_class,
        self.network.get_absolute_name_prefix(), self.name, self.output.get_description(with_name=False),
        self.construct_stack.name if self.construct_stack else None)
    else:
      return "<%s %r uninitialized, construction stack %r>" % (
        self.__class__.__name__, self.get_absolute_name(), self.construct_stack.name if self.construct_stack else None)

  def init(self, output, layer_class, template_type="template", **kwargs):
    """
    :param Data output:
    :param type[LayerBase]|LayerBase layer_class:
    :param str template_type:
    :param kwargs: via network.construct_layer, i.e. transform_config_dict was called already
    """
    output = output.copy()  # we are going to modify it here
    self.is_prev_time_frame = (template_type == "prev")
    self.is_data_template = (template_type == "template")
    assert self.is_prev_time_frame or self.is_data_template
    self.layer_class = ":%s:%s" % (template_type, layer_class.layer_class)
    self.output = output
    self.layer_class_type = layer_class
    self.kwargs = kwargs.copy()
    self.kwargs.setdefault("name", self.name)
    self.kwargs["output"] = output
    self._is_output_layer = kwargs.get("is_output_layer", None)
    self.need_last = kwargs.get("need_last", False)
    if self._has_search_choices():
      self.search_choices = SearchChoices(owner=self, beam_size=self._get_search_choices_beam_size())
    self.collocate_with = kwargs.get("collocate_with", None) or []
    self.is_initialized = True  # set last, in case there are exceptions

  def get_sub_layer(self, layer_name):
    """
    Creates a sub-layer template using self.layer_class_type.get_sub_layer_out_data_from_opts().

    :param str layer_name: name of the sub_layer (right part of '/' separated path)
    :return: template for the sub-layer
    :rtype: _TemplateLayer
    """
    full_layer_name = self.name + '/' + layer_name

    # In general, we don't know which information is needed to create the sub-layer template, so provide full kwargs
    # from the parent layer.
    res = self.layer_class_type.get_sub_layer_out_data_from_opts(layer_name, self.kwargs)
    if not res:
      raise LayerNotFound(
        "%s: Could not get out data template for sub-layer %r" % (self, layer_name),
        layer_name=full_layer_name, network=self.network)
    output, sub_layer_class, opts = res
    assert isinstance(output, Data)
    assert isinstance(sub_layer_class, type) and issubclass(sub_layer_class, LayerBase)
    assert isinstance(opts, dict)

    opts = opts.copy()
    if "network" in opts:
      assert opts.get("network") and opts.get("name"), "if network is specified, network, name must be set (%r)" % opts
    else:
      assert "name" not in opts, "name must not be set if network is not specified (%r)" % opts
      opts["network"] = self.network
      opts["name"] = full_layer_name

    opts["output"] = output
    opts["is_output_layer"] = self.is_output_layer()  # make sub-layers output layers too
    # We never get here via SubnetworkLayer but only when we explicitly need get_sub_layer,
    # so this must be collocated with the parent.
    opts["collocate_with"] = [self.name]

    # Rec sub layer GetLayer might not use this layer instance directly
    # but just copy out the template information into an own template layer instance,
    # but this should not matter here.
    sub_layer_template = _TemplateLayer(network=opts["network"], name=opts["name"])
    sub_layer_template.init(layer_class=sub_layer_class, **opts)
    return sub_layer_template

  def copy_as_prev_time_frame(self, prev_output=None, rec_vars_prev_outputs=None):
    """
    :param tf.Tensor|None prev_output:
    :param dict[str,tf.Tensor]|None rec_vars_prev_outputs:
    :return: new _TemplateLayer
    :rtype: _TemplateLayer
    """
    # Note: We assume that any dynamic sequence lengths are not changing per loop frame,
    # ie. that we can safely copy size_placeholder from the last frame.
    # However, this might not always be correct, e.g. for SliceNdLayer,
    # where you might get different sequence lengths each frame.
    # This is currently not supported.
    # We check this consistency later in RecLayer _construct.
    layer = _TemplateLayer(network=self.network, cell=self._cell, name="prev:%s" % self.name)
    layer._template_base = self
    layer.dependencies = self.dependencies
    layer.init(layer_class=self.layer_class_type, template_type="prev", **self.kwargs)
    layer.output.name = "prev:%s" % layer.output.name
    if prev_output is not None:
      layer.output.placeholder = prev_output
      layer.output.placeholder.set_shape(tf.TensorShape(layer.output.batch_shape))
      assert layer.output.placeholder.dtype is tf.as_dtype(layer.output.dtype)
    if rec_vars_prev_outputs is not None:
      layer.rec_vars_outputs = rec_vars_prev_outputs
    if layer.output.beam:
      search_choices = self.network.get_search_choices_from_beam(layer.output.beam)
      if not search_choices or search_choices.owner.network is self.network:
        layer.output.beam = layer.output.beam.copy_as_prev_frame()
        if layer.output.batch:
          layer.output.batch = layer.output.batch.copy_set_beam(layer.output.beam)
    if self.search_choices:
      layer.search_choices = SearchChoices(owner=layer, beam_size=self.search_choices.beam_size)
      if rec_vars_prev_outputs:
        layer.search_choices.set_beam_from_own_rec()
      assert layer.output.beam and layer.output.beam.beam_size == self.search_choices.beam_size
    # Note: When the sanity check fails because of missing dynamic sequence length information,
    # then the layer might dynamically construct a new sequence length, such as ConvLayer etc
    # (https://github.com/rwth-i6/returnn/wiki/Layers-which-introduce-new-dynamic-sequence-lengths).
    # In case this seq len is different per frame as mentioned in the comment above,
    # this is a problem and not supported currently.
    # In case the seq len is the same always, it just means that we did not fix the specific layer implementation yet.
    # This is work-in-progress.
    layer.output.sanity_check()
    if layer.output.placeholder is not None and self.network.get_config().bool("debug_runtime_sanity_checks", False):
      with layer.cls_setup_scope(**layer.kwargs):
        layer.output.placeholder = layer.output.get_placeholder_with_runtime_sanity_checks()
    return layer

  def _get_cell(self):
    """
    :rtype: _SubnetworkRecCell
    """
    if self._cell:
      return self._cell
    rec_layer = self.network.parent_layer
    assert isinstance(rec_layer, RecLayer)
    cell = rec_layer.cell
    assert isinstance(cell, _SubnetworkRecCell)
    return cell

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    assert self.is_initialized
    if self.is_data_template:
      # This is from the template construction, a layer in _SubnetworkRecCell.layer_data_templates.
      # Maybe we already have the layer constructed.
      real_layer = self.network.layers.get(self.name)
      if real_layer and real_layer is not self:
        return real_layer.get_dep_layers()
      # All refs to this subnet are other _TemplateLayer, no matter if prev-frame or not.
      # Otherwise, refs to the base network are given as-is.
      dependencies = list(self.cur_frame_dependencies)
      # If real layer already constructed, use it.
      dependencies = [d.get_normalized_layer() for d in dependencies]
      if self.prev_frame_dependencies:
        cell = self._get_cell()
        for layer in self.prev_frame_dependencies:
          dependencies.append(cell.get_prev_template_layer(layer.name))
      return dependencies
    assert self.is_prev_time_frame
    cell = self._get_cell()
    # In the current frame, the deps would be self.dependencies,
    # which are the logical dependencies, i.e. all such layers no matter if current or previous frame.
    # In the previous frame, just return all those dependencies, but all from previous frame.
    dependencies = []
    for layer in self.dependencies:
      if layer.network is not self.network:
        if layer not in dependencies:
          dependencies.append(layer)
        continue
      assert isinstance(layer, _TemplateLayer)
      assert layer.is_data_template
      dependencies.append(cell.get_prev_template_layer(layer.name))
    return dependencies

  def add_dependency(self, layer, is_prev_time_frame):
    """
    :param LayerBase layer:
    :param bool is_prev_time_frame:
    """
    if layer not in self.dependencies:
      self.dependencies.append(layer)
      if isinstance(layer, _TemplateLayer):
        layer.used_by.append(self)
    if is_prev_time_frame:
      assert isinstance(layer, _TemplateLayer)
      if layer not in self.prev_frame_dependencies:
        self.prev_frame_dependencies.append(layer)
    else:
      if layer not in self.cur_frame_dependencies:
        self.cur_frame_dependencies.append(layer)

  def remove_dependency(self, layer, is_prev_time_frame):
    """
    :param LayerBase layer:
    :param bool is_prev_time_frame:
    """
    self.dependencies.remove(layer)
    if isinstance(layer, _TemplateLayer):
      layer.used_by.remove(self)
    if is_prev_time_frame:
      assert isinstance(layer, _TemplateLayer)
      self.prev_frame_dependencies.remove(layer)
    else:
      self.cur_frame_dependencies.remove(layer)

  def get_normalized_layer(self):
    """
    :return: if prev layer in :class:`RecLayer`, return current layer
    :rtype: LayerBase
    """
    if self.is_prev_time_frame:
      return self._template_base.get_normalized_layer()
    if self._cell and self.name in self._cell.net.layers:
      return self._cell.net.layers[self.name].get_normalized_layer()
    return self

  def get_search_choices(self):
    """
    :rtype: SearchChoices|None
    """
    if self.search_choices:
      return self.search_choices
    if self.is_prev_time_frame:
      # Figure out search choices on current frame,
      # as dependencies can be slightly wrong.
      layer = self.get_normalized_layer()
      assert layer != self
      search_choices = layer.get_search_choices()
      if not search_choices:
        from pprint import pformat
        assert not self.output.beam, "%s: beam %r but no search choices; deps\n%s" % (
          self, self.output.beam, pformat(self.get_dep_layers()))
        return None
      if search_choices.owner.network is not self.network:  # from somewhere else...
        return search_choices
      # Normalize again. See maybe_transform.
      layer = search_choices.owner.get_normalized_layer()
      prev_layer = self._cell.net.layers["prev:%s" % layer.name]
      assert isinstance(prev_layer, _TemplateLayer) and prev_layer.is_prev_time_frame
      assert prev_layer.search_choices
      return prev_layer.search_choices
    # This is from the template construction, a layer in _SubnetworkRecCell.layer_data_templates.
    # Maybe we already have the layer constructed.
    real_layer = self.network.layers.get(self.name)
    if real_layer and real_layer is not self:
      return real_layer.get_search_choices()
    return super(_TemplateLayer, self).get_search_choices()

  def _has_search_choices(self):
    """
    :return: whether an instance of this class has search_choices set
    :rtype: bool
    """
    # TODO: extend if this is a subnet or whatever
    if issubclass(self.layer_class_type, BaseChoiceLayer):
      # Always has search_choices if we do search, even if search option is False explicitly.
      beam_size = self._get_search_choices_beam_size()
      return beam_size is not None
    return False

  def _get_search_choices_beam_size(self):
    """
    Only valid if self.has_search_choices() is True.

    :rtype: int|None
    """
    # This is usually called right after tempalte construction, but before real construction.
    # So it does not make sense to even try to get constructed real layer.
    layer_class = self.layer_class_type
    assert issubclass(layer_class, BaseChoiceLayer)
    return layer_class.cls_get_search_beam_size(**self.kwargs)

  def get_hidden_state(self):
    """
    :rtype: tf.Tensor | list[tf.Tensor] | None
    :return: optional tensor(s) with shape (time, batch, dim)
    """
    if issubclass(self.layer_class_type, RnnCellLayer):
      return self.rec_vars_outputs["state"]
    return super(_TemplateLayer, self).get_hidden_state()

  def get_last_hidden_state(self, key):
    """
    :param int|str|None key: also the special key "*"
    :rtype: tf.Tensor | None
    :return: optional tensor with shape (batch, dim)
    """
    if issubclass(self.layer_class_type, RnnCellLayer):
      return RnnCellLayer.get_state_by_key(self.rec_vars_outputs["state"], key=key)
    return super(_TemplateLayer, self).get_last_hidden_state(key=key)


class _SubnetworkRecWrappedLoss(Loss):
  """
  This wraps losses inside the loop of :class:`RecLayer`.
  """

  def __init__(self, base_loss, loss_value, error_value, norm_factor, seq_lens):
    """
    :param Loss base_loss: the loss from the layer inside the loop
    :param tf.Tensor loss_value: shape (time,batch)
    :param tf.Tensor|None error_value: shape (time,batch)
    :param tf.Tensor norm_factor: scalar for the whole batch
    :param tf.Tensor seq_lens: (batch,)
    """
    super(_SubnetworkRecWrappedLoss, self).__init__(
      base_network=base_loss.base_network,
      use_flatten_frames=base_loss.use_flatten_frames, use_normalized_loss=base_loss.use_normalized_loss,
      scale=base_loss.scale)
    assert base_loss.layer
    self.base_loss = base_loss
    self.layer = base_loss.layer  # avoid that init() gets executed again
    # Get either (time_flat,) or (time*batch,) for loss_value and error_value.
    loss_data = Data("loss", shape=(None,), time_dim_axis=0, batch_dim_axis=1, dtype="float32", placeholder=loss_value)
    loss_data.size_placeholder[0] = seq_lens
    self.loss_value = self._flatten_or_merge(loss_data)
    if error_value is not None:
      error_data = Data(
        "error", shape=(None,), time_dim_axis=0, batch_dim_axis=1, dtype=error_value.dtype.name,
        placeholder=error_value)
      error_data.size_placeholder[0] = seq_lens
      self.error_value = self._flatten_or_merge(error_data)
    else:
      self.error_value = None  # type: typing.Optional[tf.Tensor]
    self.loss_norm_factor = norm_factor

  def init(self, output, output_with_activation=None, target=None, layer=None):
    """
    :param Data output:
    :param None|returnn.tf.layers.basic.OutputWithActivation output_with_activation:
    :param Data|None target:
    :param LayerBase|None layer:
    """
    self.output = output
    self.layer = layer
    # ignore otherwise

  def get_value(self):
    """
    :rtype: tf.Tensor
    """
    return self.reduce_func(self.loss_value)

  def get_error(self):
    """
    :rtype: tf.Tensor|None
    """
    if self.error_value is not None:
      return self.reduce_func(self.error_value)
    else:
      return None


class RecStepInfoLayer(LayerBase):
  """
  Used by _SubnetworkRecCell.
  Represents the current step number.
  Usually via :func:`TFNetwork.set_rec_step_info`.
  """

  layer_class = ":i"

  def __init__(self, i=None, prev_end_flag=None, prev_end_layer=None, seq_lens=None, **kwargs):
    """
    :param tf.Tensor|None i: scalar, int32, current step (time)
    :param tf.Tensor|None prev_end_flag: (batch,), bool, says that the current sequence has ended.
      Can be with beam. In that case, end_flag_source should be "prev:end", and define the search choices.
    :param LayerBase|None prev_end_layer: corresponds to the "prev:end" layer if available
    :param tf.Tensor|None seq_lens: (batch,) int32, seq lens
    """
    if "output" not in kwargs:
      kwargs = kwargs.copy()
      kwargs["output"] = self.get_out_data_from_opts(network=kwargs["network"])
    super(RecStepInfoLayer, self).__init__(**kwargs)
    self.step = None
    self._end_flag = None
    self.prev_end_layer = None
    if not self.output.have_time_axis():  # the normal case
      assert i is not None and i.get_shape().ndims == 0
      self.output.placeholder = i
      self.step = i
      self._end_flag = prev_end_flag
      self.prev_end_layer = prev_end_layer
    else:
      # This only is valid if we are moved out from a RecLayer.
      assert self.output.size_placeholder and 0 in self.output.size_placeholder
      seq_lens = self.output.size_placeholder[0]
      self.output.placeholder = tf.range(tf.reduce_max(seq_lens))
    self._seq_lens = seq_lens
    if seq_lens is None:
      assert prev_end_layer

  def get_prev_end_flag(self, target_search_choices):
    """
    :param SearchChoices|None target_search_choices:
    :return: (batch,) of type bool. batch might include beam size.
      This returns the end flag corresponding to the last frame.
      I.e. if the "end" layer exists and is used, this is "prev:end".
    :rtype: tf.Tensor
    """
    if self._end_flag is not None:
      end_flag = self._end_flag
    else:
      assert self._seq_lens is not None
      from returnn.tf.util.basic import reuse_name_scope_of_tensor
      with reuse_name_scope_of_tensor(self.step, postfix="/end_flag"):
        rec_layer = self.network.get_rec_parent_layer()
        assert rec_layer
        end_flag = tf.greater_equal(
          # Without include_eos, end_flag=True happens first in frame seq_lens.
          # With include_eos, end_flag=True happens first in frame seq_lens - 1.
          # We are supposed to return the prev end flag here.
          self.step,
          self._seq_lens if rec_layer.include_eos else (self._seq_lens + 1))
    if self.prev_end_layer:
      source_search_choices = self.prev_end_layer.get_search_choices()
    else:
      # assume same search choices, same beam
      return end_flag
    if target_search_choices:
      if source_search_choices:
        assert self.prev_end_layer
        end_flag_transformed_layer = target_search_choices.translate_to_this_search_beam(self.prev_end_layer)
        assert isinstance(end_flag_transformed_layer, LayerBase)
        end_flag = end_flag_transformed_layer.output.placeholder
      else:
        from returnn.tf.util.basic import tile_transposed
        end_flag = tile_transposed(end_flag, axis=0, multiples=target_search_choices.beam_size)
    else:
      assert not self.prev_end_layer or not source_search_choices
    return end_flag

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    d.setdefault("from", [])  # source does not make sense
    super(RecStepInfoLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

  @classmethod
  def get_out_data_from_opts(cls, network, **kwargs):
    """
    :param returnn.tf.network.TFNetwork network:
    :rtype: Data
    """
    # Check for the normal case first. If we don't have a parent rec layer, also fallback to this (e.g. debugging).
    if network.is_inside_rec_layer() or not isinstance(network.parent_layer, RecLayer):
      return Data(name="i", shape=(), batch_dim_axis=None, dtype="int32", sparse=False)
    # This only is valid if we are moved out from a RecLayer.
    assert isinstance(network.parent_layer, RecLayer)
    # We need to get the time-dim and seq lens.
    # Maybe this is not the best way.
    # But we could extend _SubnetworkRecCell later to get this more directly if needed.
    assert 0 in network.parent_layer.output.size_placeholder
    seq_lens = network.parent_layer.output.size_placeholder[0]
    return Data(
      name="i_unrolled", shape=(None,), time_dim_axis=0, batch_dim_axis=None, dtype="int32", sparse=False,
      size_placeholder={0: seq_lens})


class RecLastOutputLayer(LayerBase):
  """
  Gets the last output from some sub layer inside a :class:`RecLayer`.
  You should explicitly set ``need_last`` on the specific layer such that this information is available.
  """
  layer_class = "rec_last_output"

  def __init__(self, rec_layer, sub_layer_name, **kwargs):
    """
    :param RecLayer rec_layer:
    :param str sub_layer_name:
    """
    super(RecLastOutputLayer, self).__init__(**kwargs)
    rec_layer = self._resolve_to_rec_layer(rec_layer, repr(self))
    cell = rec_layer.cell
    assert isinstance(cell, _SubnetworkRecCell)
    out = cell.get_layer_last_frame(sub_layer_name)
    assert out.dim_tags == self.output.dim_tags
    self.output.placeholder = out.placeholder

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    d.setdefault("from", ())
    super(RecLastOutputLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["rec_layer"] = get_layer(d["rec_layer"])

  @classmethod
  def get_out_data_from_opts(cls, rec_layer, sub_layer_name, name, **kwargs):
    """
    :param RecLayer rec_layer:
    :param str sub_layer_name:
    :param str name:
    :rtype: Data
    """
    if isinstance(rec_layer, _TemplateLayer):
      assert issubclass(rec_layer.layer_class_type, RecLayer)
      cell = rec_layer.kwargs["unit"]
    else:
      rec_layer = cls._resolve_to_rec_layer(rec_layer, "%s %r" % (cls.__name__, name))
      cell = rec_layer.cell
    assert isinstance(cell, _SubnetworkRecCell)
    return cell.layer_data_templates[sub_layer_name].output

  @classmethod
  def _resolve_to_rec_layer(cls, layer, exception_prefix):
    """
    :param RecLayer|LayerBase layer:
    :param str exception_prefix:
    :rtype: RecLayer
    """
    from returnn.tf.layers.basic import SubnetworkLayer, CopyLayer
    while not isinstance(layer, RecLayer):
      if isinstance(layer, SubnetworkLayer):
        layer = layer.subnetwork.get_layer("output")
      elif isinstance(layer, CopyLayer) and len(layer.sources) == 1:
        layer = layer.sources[0]
      else:
        raise Exception("%s: layer %r is not a RecLayer" % (exception_prefix, layer))
    return layer


class RnnCellLayer(_ConcatInputLayer):
  """
  Wrapper around tf.contrib.rnn.RNNCell.
  This will operate a single step, i.e. there is no time dimension,
  i.e. we expect a (batch,n_in) input, and our output is (batch,n_out).
  This is expected to be used inside a RecLayer.
  (But it can also handle the case to be optimized out of the rec loop,
   i.e. outside a RecLayer, with a time dimension.)
  """

  layer_class = "rnn_cell"
  recurrent = True

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
    from returnn.tf.util.basic import get_initializer
    scope = "rec" if self.name_scope is None else tf_compat.v1.get_variable_scope()
    with reuse_name_scope(scope), self.var_creation_scope(
      initializer=get_initializer(
        weights_init, seed=self.network.random.randint(2 ** 31), eval_local_ns={"layer": self})
    ) as scope:
      assert isinstance(scope, tf_compat.v1.VariableScope)
      scope_name_prefix = scope.name + "/"  # e.g. "layer1/rec/"
      self.cell = self._get_cell(n_out=n_out, unit=unit, unit_opts=unit_opts)
      assert isinstance(self.cell, rnn_cell.RNNCell)
      if self._rec_previous_layer:
        x = self.input_data.placeholder
        if isinstance(self.cell, BaseRNNCell):
          x = self.cell.get_input_transformed(x)
        assert not self.input_data or self.input_data.time_dim_axis is None, (
          self, self.input_data,
          "A recurrent layer is not allowed to have input data with a remaining time axis.\n"
          "A possible reason for this error is that the 'target' of the rec layer does not\n"
          "match the targets of the sub-layers")
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
        self.output.placeholder, state = tf_compat.v1.nn.dynamic_rnn(
          self.cell,
          inputs=x,
          sequence_length=self.input_data.get_sequence_lengths(),
          initial_state=state0, time_major=True, scope=scope)
      self._hidden_state = state
      self.rec_vars_outputs["state"] = state
      params = tf_compat.v1.get_collection(
        tf_compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=re.escape(scope_name_prefix))
      for p in params:
        self.add_param(p)

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
    rnn_cell_class = RecLayer.get_rnn_cell_class(unit, cell_only=True)
    # E.g. rnn_cell_class is :class:`rnn_cell.LSTMCell`.
    if issubclass(rnn_cell_class, rnn_cell.RNNCell):
      if unit_opts is None:
        unit_opts = {}
      assert isinstance(unit_opts, dict)
      # This should not have any side-effects, i.e. it should not add to the current computation graph,
      # it should also not create any vars yet, etc.
      # noinspection PyArgumentList
      cell = rnn_cell_class(n_out, **unit_opts)
      assert isinstance(cell, rnn_cell.RNNCell)
      return cell
    import returnn.tf.native_op as tf_native_op
    if issubclass(rnn_cell_class, tf_native_op.RecSeqCellOp):
      # noinspection PyArgumentList
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
    sources_data = Data.get_common_data([src.output for src in sources if src], ignore_feature_dim=True)
    batch_dim = sources_data.get_batch_dim_tag() if sources_data and sources_data.have_batch_axis() else None
    time_dim_axis, batch_dim_axis = None, None
    dim_tags = []
    if sources_data and sources_data.have_time_axis():
      time_dim_axis = len(dim_tags)
      dim_tags.append(sources_data.get_time_dim_tag())
    if batch_dim:
      batch_dim_axis = len(dim_tags)
      dim_tags.append(batch_dim)
    dim_tags.append(FeatureDim("%s:rnn_cell_feat" % name, dimension=n_out, auto_generated=True))
    return Data(
      name="%s_output" % name,
      dim_tags=dim_tags, dim=n_out,
      batch_dim_axis=batch_dim_axis,
      time_dim_axis=time_dim_axis,
      batch=sources_data.batch,
      beam=sources_data.beam)

  def get_absolute_name_scope_prefix(self):
    """
    :rtype: str
    """
    if self.name_scope is not None:
      return self.get_base_absolute_name_scope_prefix()
    return self.get_base_absolute_name_scope_prefix() + "rec/"  # all under "rec" sub-name-scope

  def get_dep_layers(self):
    """
    :rtype: list[tf.Tensor]
    """
    ls = list(super(RnnCellLayer, self).get_dep_layers())

    def visit(s):
      """
      :param list|tuple|dict|LayerBase|str|int|float|None s:
      """
      if isinstance(s, (list, tuple)):
        for x in s:
          visit(x)
      elif isinstance(s, dict):
        for x in s.values():
          visit(x)
      elif isinstance(s, LayerBase):
        ls.append(s)
      else:
        assert isinstance(s, (str, int, float, type(None)))

    visit(self._initial_state)
    return ls

  # noinspection PyUnusedLocal
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

  # noinspection PyUnusedLocal
  @classmethod
  def get_output_from_state(cls, state, unit):
    """
    :param tuple[tf.Tensor]|tf.Tensor state:
    :param str unit:
    :rtype: tf.Tensor
    """
    if isinstance(state, rnn_cell.LSTMStateTuple):
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
  def get_state_by_key(cls, state, key, shape=None):
    """
    :param tf.Tensor|tuple[tf.Tensor]|namedtuple state:
    :param int|str|None key:
    :param tuple[int|None] shape: Shape of the state.
    :rtype: tf.Tensor
    """
    from tensorflow.python.util import nest
    from returnn.util.basic import is_namedtuple
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
    if shape is None:
      x.set_shape(tf.TensorShape([None, None]))  # Assume (batch,dim).
    else:
      x.set_shape(tf.TensorShape([None] * len(shape)))
    return x

  def get_last_hidden_state(self, key):
    """
    :param int|str|None key:
    :rtype: tf.Tensor
    """
    return self.get_state_by_key(self._hidden_state, key=key)

  @classmethod
  def get_rec_initial_state(cls, batch_dim, name, unit,
                            n_out=None, out_dim=None,
                            initial_state=None, unit_opts=None,
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
    :param int|None n_out: out dim
    :param Dim|None out_dim: out dim
    :param str unit: cell name
    :param dict[str]|None unit_opts:
    :param LayerBase|str|int|float|None|list|tuple|namedtuple initial_state: see code
    :param RecLayer|LayerBase|None rec_layer: for the scope
    :rtype: tf.Tensor|tuple[tf.Tensor]|namedtuple
    """
    if n_out is None:
      assert out_dim is not None and out_dim.dimension is not None
      n_out = out_dim.dimension
    with tf.name_scope("rec_initial_state"):
      init_value = initial_state
      dim = cls.get_hidden_state_size(n_out=n_out, unit=unit, unit_opts=unit_opts, **kwargs)

      # noinspection PyShadowingNames
      def make_list(keys):
        """
        :param list[str|int]|tuple[str|int] keys:
        :rtype: list[tf.Tensor]
        """
        assert isinstance(keys, (tuple, list))
        assert len(keys) == len(dim)
        if isinstance(init_value, (list, tuple)):
          assert len(init_value) == len(dim)
          return [cls.get_rec_initial_state_inner(initial_shape=(batch_dim, d), initial_state=v_, key=k, name=name,
                                                  rec_layer=rec_layer) for (d, v_, k) in zip(dim, init_value, keys)]
        # Do not broadcast LayerBase automatically in this case.
        assert isinstance(init_value, (int, float, str, type(None)))
        return [cls.get_rec_initial_state_inner(initial_shape=(batch_dim, d), initial_state=init_value, key=k,
                                                name=name, rec_layer=rec_layer) for d, k in zip(dim, keys)]

      # Make it the same type because nest.assert_same_structure() will complain otherwise.
      from returnn.util.basic import is_namedtuple
      if is_namedtuple(type(dim)):
        # noinspection PyProtectedMember,PyUnresolvedReferences
        keys = dim._fields
        assert len(dim) == len(keys)
        assert isinstance(init_value, (int, float, str, tuple, list, dict, type(None)))
        if not isinstance(init_value, dict) and init_value not in (0, 1, None) and not isinstance(init_value, str):
          print(("Layer %r: It is recommended to use a dict to specify 'initial_state'"
                 "with keys %r for the state dimensions %r.") % (name, keys, dim), file=log.v2)
        if isinstance(init_value, dict):
          assert set(init_value.keys()) == set(keys), "You must specify all keys for the state dimensions %r." % dim
          assert len(init_value) == len(dim)
          s = {k: cls.get_rec_initial_state_inner(initial_shape=(batch_dim, d), initial_state=init_value[k], key=k,
                                                  name=name, rec_layer=rec_layer) for (k, d) in zip(keys, dim)}
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
        return cls.get_rec_initial_state_inner(initial_shape=(batch_dim, dim), initial_state=init_value, key=None,
                                               name=name, rec_layer=rec_layer)
      else:
        raise Exception("Did not expect hidden_state_size %r." % dim)

  @classmethod
  def get_rec_initial_state_inner(cls, initial_shape, name, state_key='state', key=None, initial_state=None,
                                  shape_invariant=None, rec_layer=None):
    """
    Generate initial hidden state. Primarily used as a inner function for RnnCellLayer.get_rec_initial_state.

    :param tuple initial_shape: shape of the initial state.
    :param str name: layer name.
    :param str state_key: key to be used to get the state from final_rec_vars.
    :param str|None key: key/attribute of the state if state is a dictionary/namedtuple
      (like 'c' and 'h' for LSTM states).
    :param LayerBase|str|int|float|None|list|tuple|namedtuple initial_state: see code
    :param tuple shape_invariant: If provided, directly used. Otherwise, guessed from initial_shape (see code below).
    :param RecLayer|LayerBase|None rec_layer: For the scope.
    :rtype: tf.Tensor
    """
    key_name = str(key if key is not None else "var")
    from returnn.util.basic import dummy_noop_ctx
    if shape_invariant is None:
      shape_invariant = tuple([d if isinstance(d, int) and d != 0 else None for d in initial_shape])
    if isinstance(initial_state, LayerBase):
      h = initial_state.get_last_hidden_state(key="*")
      if h is not None:
        h.set_shape(shape_invariant)
        return h
      assert initial_state.output.batch_dim_axis == 0
      assert initial_state.output.time_dim_axis is None
      assert initial_state.output.shape == initial_shape[1:]
      assert initial_state.output.placeholder is not None
      return initial_state.output.placeholder
    elif initial_state == "zeros" or not initial_state:
      return tf.zeros(initial_shape)
    elif initial_state == "ones" or initial_state == 1:
      return tf.ones(initial_shape)
    elif initial_state == "var":  # Initial state is a trainable variable.
      # Assume the first dimension to be batch_dim.
      assert shape_invariant[0] is None and all([d is not None for d in shape_invariant[1:]])
      with rec_layer.var_creation_scope() if rec_layer else dummy_noop_ctx():
        var = tf_compat.v1.get_variable(
          "initial_%s" % key_name, shape=initial_shape[1:], initializer=tf.zeros_initializer())
      from returnn.tf.util.basic import expand_dims_unbroadcast
      var = expand_dims_unbroadcast(var, axis=0, dim=initial_shape[0])  # (batch,dim)
      return var
    elif initial_state == "keep_over_epoch" or initial_state == "keep_over_epoch_no_init":
      # "keep_over_epoch_no_init" should only be used to build a graph for use outside returnn.
      assert rec_layer is not None
      with rec_layer.var_creation_scope():
        var = tf_compat.v1.get_variable(
          'keep_state_%s' % key_name,
          validate_shape=False, initializer=tf_util.zeros_dyn_shape(shape_invariant),  # Dummy state, will not be used.
          trainable=False, collections=[tf_compat.v1.GraphKeys.GLOBAL_VARIABLES, tf_util.CollectionKeys.STATE_VARS])
      assert isinstance(var, tf.Variable)
      var.set_shape(shape_invariant)
      rec_layer.saveable_param_replace[var] = None  # Do not save this variable.

      def update_var():
        """
        :return: nothing, calls :func:`TFNetwork.register_post_control_dependencies`.
        """
        if isinstance(rec_layer, RecLayer) and isinstance(rec_layer.cell, _SubnetworkRecCell):
          final_rec_vars = rec_layer.cell.get_final_rec_vars(name)
          last_state = cls.get_state_by_key(final_rec_vars[state_key], key=key, shape=initial_shape)
        else:
          last_state = rec_layer.get_last_hidden_state(key=key)
        last_state.set_shape(shape_invariant)
        rec_layer.network.register_post_control_dependencies(
          [tf_compat.v1.assert_equal(tf.rank(last_state), len(shape_invariant), name="check_last_state_rank")] +
          [tf_compat.v1.assert_equal(tf.shape(last_state)[i], dim, name="check_last_state_dim_%i" % i)
           for i, dim in enumerate(shape_invariant) if dim is not None] +
          [tf_compat.v1.assign(var, last_state, validate_shape=False, name="assign_last_state")])

      rec_layer.post_init_hooks.append(update_var)
      if initial_state == "keep_over_epoch_no_init":
        return var.value()
      else:
        step = rec_layer.network.get_epoch_step()
        # Note: If you get somewhere an error like `In[0] is not a matrix` or so,
        # likely `update_var` was not correctly called or handled.
        s = tf.cond(tf.equal(step, 0), lambda: tf.zeros(initial_shape), lambda: var.value())
        s.set_shape(shape_invariant)
        return s
    else:
      raise Exception("invalid initial state type %r for sub-layer %r, key %r" % (initial_state, name, key))

  @classmethod
  def get_rec_initial_extra_outputs(cls, **kwargs):
    """
    :rtype: dict[str,tf.Tensor|tuple[tf.Tensor]]
    """
    return {"state": cls.get_rec_initial_state(**kwargs)}

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    super(RnnCellLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if "initial_state" in d:
      d["initial_state"] = cls.transform_initial_state(d["initial_state"], network=network, get_layer=get_layer)

  # noinspection PyUnusedLocal
  @staticmethod
  def transform_initial_state(initial_state, network, get_layer):
    """
    :param str|float|int|list[str|float|int]|dict[str]|None initial_state:
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    def resolve(v):
      """
      :param str|tuple|list|float|int|None v:
      :return:
      """
      if isinstance(v, str):
        if v in ["zeros", "ones", "var", "keep_over_epoch", "keep_over_epoch_no_init"]:
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
    """
    :param str unit:
    :param None initial_output:
    :param LayerBase|str|int|float|None|list|tuple|namedtuple initial_state:
    :rtype: tf.Tensor
    """
    from tensorflow.python.util import nest
    if all(v in [None, 0, "zeros"] for v in nest.flatten(initial_state)):
      assert initial_output in (None, 0), "layer %r: use initial_state instead" % kwargs["name"]
      # We can just return 0.
      return super(RnnCellLayer, cls).get_rec_initial_output(initial_output=0, **kwargs)
    assert initial_output is None, "layer %r: use initial_state instead" % kwargs["name"]
    state = cls.get_rec_initial_state(unit=unit, initial_state=initial_state, **kwargs)
    return cls.get_output_from_state(state=state, unit=unit)


class GetLastHiddenStateLayer(LayerBase):
  """
  Will combine (concat or add or so) all the last hidden states from all sources.
  """

  layer_class = "get_last_hidden_state"

  def __init__(self, out_dim=None, n_out=None, combine="concat", key='*', **kwargs):
    """
    :param Dim|None out_dim:
    :param int|None n_out: dimension. output will be of shape (batch, n_out)
    :param str combine: "concat" or "add"
    :param str|int|None key: for the state, which could be a namedtuple. see :func:`RnnCellLayer.get_state_by_key`
    """
    if not n_out:
      assert out_dim
      n_out = out_dim.dimension
    if n_out and out_dim:
      assert n_out == out_dim.dimension
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
    from returnn.tf.util.basic import check_input_ndim, check_input_dim
    h = check_input_ndim(h, 2)
    h = check_input_dim(h, 1, n_out)
    self.output.placeholder = h

  def get_last_hidden_state(self, key):
    """
    :param str|None key:
    :rtype: tf.Tensor
    """
    assert key in [None, '*']
    return self.output.placeholder

  @classmethod
  def get_out_data_from_opts(cls, name, sources, out_dim=None, n_out=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param Dim|None out_dim:
    :param int|None n_out: dimension. output will be of shape (batch, n_out)
    :rtype: Data
    """
    from returnn.tf.util.data import batch_dim
    if not out_dim:
      assert n_out
      out_dim = FeatureDim("%s:hidden-out" % name, n_out, auto_generated=True)
    out = Data("%s_output" % name, dim_tags=[batch_dim, out_dim])
    out.beam = sources[0].output.beam
    out.batch = sources[0].output.batch
    return out


class GetRecAccumulatedOutputLayer(LayerBase):
  """
  For :class:`RecLayer` with a subnet.
  If some layer is explicitly marked as an additional output layer (via 'is_output_layer': True),
  you can get that subnet layer output via this accessor.
  Retrieves the accumulated output.

  Note that this functionality is obsolete now. You can simply access such an sub layer
  via the generic sub layer access mechanism. I.e. instead of::

    "sub_layer": {"class": "get_rec_accumulated", "from": "rec_layer", "sub_layer": "hidden"}

  You can do::

    "sub_layer": {"class": "copy", "from": "rec_layer/hidden"}
  """
  layer_class = "get_rec_accumulated"

  # noinspection PyUnusedLocal
  def __init__(self, sub_layer, **kwargs):
    """
    :param str sub_layer: layer of subnet in RecLayer source, which has 'is_output_layer': True
    """
    super(GetRecAccumulatedOutputLayer, self).__init__(**kwargs)
    # Nothing needs to be done, all logic in self.get_out_data_from_opts already.

  @classmethod
  def get_out_data_from_opts(cls, name, sources, sub_layer, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str sub_layer:
    :rtype: Data
    """
    assert len(sources) == 1, "%s %r: expect exactly one source" % (cls, name)
    rec_layer = sources[0]
    assert isinstance(rec_layer, RecLayer), "%s %r: expect that the source is a RecLayer" % (cls, name)
    assert isinstance(rec_layer.cell, _SubnetworkRecCell), "%s %r: expect a RecLayer with subnet" % (cls, name)
    assert rec_layer.cell.output_layers_net, "%s %r: expect a RecLayer with output net" % (cls, name)
    subnet = rec_layer.cell.output_layers_net
    assert sub_layer in subnet.layers, "%s %r: maybe %r not with 'is_output_layer'?" % (
      cls, name, sub_layer)
    return subnet.layers[sub_layer].output


class RecUnstackLayer(LayerBase):
  """
  This is supposed to be used inside a :class:`RecLayer`.
  The input is supposed to be outside the rec layer (i.e. via ``base:``).
  Uses tf.TensorArray and then unstack on the inputs to make it available per-frame.
  This is an alternative to making some input to the rec layer,
  such that the rec layer can have multiple inputs (as long as they have the same time dim).

  Note that due to automatic optimization, this layer will be optimized out of the rec loop anyway,
  and then the tf.TensorArray logic happens internally in RecLayer,
  thus we do not need to care about this here.
  (See get_input_moved_out for some internal handling.)

  Effectively, this layer is very similar to :class:`CopyLayer`,
  with the only special behavior that it checks (or even assigns) the loop dimension of RecLayer.
  """
  layer_class = "rec_unstack"

  def __init__(self, axis=None, declare_rec_time=False, **kwargs):
    """
    Due to automatic optimization, not much happens here.
    The real logic happens in :func:`get_out_data_from_opts`.

    Note that it is allowed to leave both `axis` and `declare_rec_time` unset,
    in case you assign `axis` to the rec layer, and the source here has the same axis (dim tag).

    :param str|Dim|None axis:
    :param bool declare_rec_time:
    """
    axis, declare_rec_time  # noqa  # unused here, used in get_out_data_from_opts
    super(RecUnstackLayer, self).__init__(**kwargs)
    assert len(self.sources) == 1
    src = self.sources[0].output
    rec_time_dim = self.network.get_inside_rec_time_dim()
    if rec_time_dim:
      raise NotImplementedError("%s: We expect that this layer is always optimized out." % self)
    self.output.placeholder = src.placeholder

  @classmethod
  def get_out_data_from_opts(cls, name, sources, network, axis=None, declare_rec_time=False, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param returnn.tf.network.TFNetwork network:
    :param str|Dim|None axis:
    :param bool declare_rec_time:
    :rtype: Data
    """
    assert sources
    out = sources[0].output.copy_template(name="%s_output" % name)
    if not axis:
      axis = network.get_inside_rec_time_dim(inside_loop=False)
      assert axis, "%s %r: expected to be inside rec layer" % (cls.__name__, name)
    out_dim = out.get_dim_tag_from_description(axis)
    rec_time_dim = network.get_inside_rec_time_dim()
    if rec_time_dim:
      if rec_time_dim.is_dim_known():  # defined
        assert out_dim == rec_time_dim, "%s %r: rec dim %s does not match input %s dim %s" % (
          cls.__name__, name, rec_time_dim, sources[0], out_dim)
      else:
        if out_dim.is_dim_known():  # usually the case except at template construction
          assert out_dim != rec_time_dim  # rec_time_dim is unknown, so it cannot be the same
        if out_dim != rec_time_dim:
          assert declare_rec_time, "%s %r: must either set known axis on rec %s or enable declare_rec_time" % (
            cls.__name__, name, rec_time_dim)
          rec_time_dim.declare_same_as(out_dim)
      out.mark_same_time(out_dim, must_match=True)
      return out.copy_template_excluding_time_dim()
    else:
      out.mark_same_time(out_dim, must_match=True)
      return out


class BaseChoiceLayer(LayerBase):
  """
  This is a base-class for any layer which defines a new search choice,
  i.e. which defines ``self.search_choices``.
  """

  # noinspection PyUnusedLocal
  def __init__(self, beam_size, search=NotSpecified, add_to_beam_scores=NotSpecified, **kwargs):
    """
    :param int|None beam_size: the outgoing beam size. i.e. our output will be (batch * beam_size, ...)
    :param NotSpecified|bool search: whether to perform search, or use the ground truth (`target` option).
      If not specified, it will depend on `network.search_flag`.
    :param NotSpecified|bool add_to_beam_scores: whether to add the scores to the beam scores.
      This will be done with search obviously (not supported to not do it).
      Without search, we can still add the scores of the ground-truth labels to the beam.
      By default, this is derived from `search or network.search_flag`.
      So with enabled net search flag, even when `search` is disabled here, it will add the scores.
    """
    super(BaseChoiceLayer, self).__init__(**kwargs)

  # noinspection PyUnusedLocal
  @classmethod
  def cls_get_search_beam_size(
        cls, network, beam_size, search=NotSpecified, add_to_beam_scores=NotSpecified,
        sources=(), _src_common_search_choices=None, **kwargs):
    """
    :param returnn.tf.network.TFNetwork network:
    :param list[LayerBase] sources:
    :param int|None beam_size: the outgoing beam size. i.e. our output will be (batch * beam_size, ...)
    :param NotSpecified|bool search:
    :param NotSpecified|bool add_to_beam_scores:
    :param None|SearchChoices _src_common_search_choices: set via :func:`SearchChoices.translate_to_common_search_beam`
    :return: when this layer provides an own choice (search_choices attrib is set), then the corresponding beam size
    :rtype: int|None
    """
    search = NotSpecified.resolve(search, network.search_flag)
    add_to_beam_scores = NotSpecified.resolve(add_to_beam_scores, search or network.search_flag)
    if not search and not add_to_beam_scores:
      return None
    return beam_size

  # noinspection PyMethodOverriding
  @classmethod
  def get_rec_initial_extra_outputs(cls, network, beam_size, search=NotSpecified, add_to_beam_scores=NotSpecified,
                                    **kwargs):
    """
    :param returnn.tf.network.TFNetwork network:
    :param int beam_size:
    :param NotSpecified|bool search:
    :param NotSpecified|bool add_to_beam_scores:
    :rtype: dict[str,tf.Tensor]
    """
    search = NotSpecified.resolve(search, network.search_flag)
    add_to_beam_scores = NotSpecified.resolve(add_to_beam_scores, search or network.search_flag)
    if not search and not add_to_beam_scores:
      return {}
    batch_dim = network.get_data_batch_dim()
    # Note: Use beam_size 1 for the initial as there are no competing hypotheses yet.
    initial_scores = tf.zeros([batch_dim, 1])  # (batch, beam)
    # However! Our initial output is *with* the beam size, and SelectSearchSourcesLayer should keep the beam size.
    # We just use 0s, as we don't really know the incoming beam size at this point,
    # and it should not matter anyway.
    initial_src_beams = tf.zeros([batch_dim, beam_size], dtype=tf.int32)
    # Note: Our rec vars are handled via SearchChoices.set_beam_scores.
    return {"choice_scores": initial_scores, "choice_src_beams": initial_src_beams}

  @classmethod
  def get_rec_initial_extra_outputs_shape_invariants(cls, **kwargs):
    """
    :rtype: dict[str,tf.TensorShape]
    """
    # Initial beam size is 1 and then later the given one, so it changes.
    return {
      "choice_scores": tf.TensorShape((None, None)),  # (batch, beam)
      "choice_src_beams": tf.TensorShape((None, None)),  # (batch, beam)
    }

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    super(BaseChoiceLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if "rec_previous_layer" in d:
      prev_layer = d["rec_previous_layer"]
      assert isinstance(prev_layer, _TemplateLayer)
      assert prev_layer.is_prev_time_frame
      # Note: In SearchChoices.translate_to_common_search_beam, we would get this prev_layer.
      # And we might not be able to compare it to other search choices,
      # as it might not be in the search choices sequence.
      # But we actually do not care at all about it, and do not use it. So just reset.
      d["rec_previous_layer"] = None


class ChoiceLayer(BaseChoiceLayer):
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

  In case of multiple sources, this layer computes the top-k combinations of choices. The score of such a combination
  is determined by adding up the (log-space) scores of the choices for the individual sources. In this case, the
  'target' parameter of the layer has to be set to a list of targets corresponding to the sources respectively. Because
  computing all possible combinations of source scores is costly, the sources are pruned beforehand using the beam
  sizes set by the 'source_beam_sizes' parameter. The choices made for the different sources can be accessed via the
  sublayers '<choice layer name>/out_0', '<choice layer name>/out_1' and so on.
  Note, that the way scores are combined assumes the sources to be independent. If you want to model a dependency,
  use separate ChoiceLayers and let the input of one depend on the output of the other.
  """
  layer_class = "choice"

  _debug_out = None  # type: typing.Optional[list]

  def __init__(self, beam_size, keep_beams=False,
               search=NotSpecified,
               add_to_beam_scores=NotSpecified,
               input_type="prob",
               prob_scale=1.0, base_beam_score_scale=1.0, random_sample_scale=0.0,
               length_normalization=True,
               length_normalization_exponent=1.0,
               custom_score_combine=None,
               source_beam_sizes=None, scheduled_sampling=False, cheating=False,
               explicit_search_sources=None,
               **kwargs):
    """
    :param int beam_size: the outgoing beam size. i.e. our output will be (batch * beam_size, ...)
    :param bool keep_beams: specifies that we keep the beam_in entries,
      i.e. we just expand, i.e. we just search on the dim. beam_size must be a multiple of beam_in.
    :param NotSpecified|bool search: whether to perform search, or use the ground truth (`target` option).
      If not specified, it will depend on `network.search_flag`.
    :param NotSpecified|bool add_to_beam_scores: whether to add the scores to the beam scores.
      This will be done with search obviously (not supported to not do it).
      Without search, we can still add the scores of the ground-truth labels to the beam.
      By default, this is derived from `search or network.search_flag`.
      So with enabled net search flag, even when `search` is disabled here, it will add the scores.
    :param str input_type: "prob", "log_prob" or "logits", whether the input is in probability space, log-space, etc.
      or "regression", if it is a prediction of the data as-is. If there are several inputs, same format
      for all is assumed.
    :param float prob_scale: factor for prob (score in +log space from source)
    :param float base_beam_score_scale: factor for beam base score (i.e. prev prob scores)
    :param float random_sample_scale: if >0, will add Gumbel scores. you might want to set base_beam_score_scale=0
    :param bool length_normalization: evaluates score_t/len in search
    :param list[int]|None source_beam_sizes: If there are several sources, they are pruned with these beam sizes
       before combination. If None, 'beam_size' is used for all sources. Has to have same length as number of sources.
    :param dict|None scheduled_sampling:
    :param bool|str cheating: if True, will always add the true target in the beam.
      if "exclusive", enables cheating_exclusive. see :func:`returnn.tf.util.basic.beam_search`.
    :param list[LayerBase]|None explicit_search_sources: will mark it as an additional dependency.
      You might use these also in custom_score_combine.
    :param callable|None custom_score_combine:
    """
    super(ChoiceLayer, self).__init__(beam_size=beam_size, search=search, **kwargs)
    from returnn.util.basic import CollectionReadCheckCovered
    from returnn.tf.util.basic import optional_add, optional_mul, batch_gather, expand_dims_unbroadcast
    search = NotSpecified.resolve(search, default=self.network.search_flag)
    assert isinstance(search, bool)
    add_to_beam_scores = NotSpecified.resolve(add_to_beam_scores, default=search or self.network.search_flag)
    self.search_flag = search
    self.input_type = input_type
    self.length_normalization = length_normalization
    self.explicit_search_sources = explicit_search_sources
    self.scheduled_sampling = CollectionReadCheckCovered.from_bool_or_dict(scheduled_sampling)
    self.cheating = cheating
    self.search_scores_in = None
    self.search_scores_base = None
    self.search_scores_combined = None
    # We assume log-softmax here, inside the rec layer.

    if self.search_flag:
      if cheating:
        print("%s: cheating %r, i.e. we add the ground truth to the beam" % (self, cheating), file=log.v2)
      for source in self.sources:
        assert not source.output.sparse
      assert self.sources[0].output.dim == self.output.dim
      assert self.sources[0].output.shape == (self.output.dim,)

      # We are doing the search.
      self.search_choices = SearchChoices(
        owner=self,
        beam_size=beam_size)
      if input_type == "regression":
        assert len(self.sources) == 1
        # It's not a probability distribution, so there is no search here.
        net_batch_dim = self.network.get_data_batch_dim()
        assert self.search_choices.beam_size == 1
        assert not cheating
        self.output = self.sources[0].output.copy_compatible_to(self.output)
        self.search_choices.set_src_beams(tf.zeros((net_batch_dim, 1), dtype=tf.int32))
        self.search_choices.set_beam_scores(self.search_choices.src_layer.search_choices.beam_scores)
      else:
        net_batch_dim = self.network.get_data_batch_dim()
        if len(self.sources) > 1:
          # If no separate beam sizes for the sources are given, use the final beam size also for pruning
          # the incoming sources. Note, that it makes no sense to set it higher than that, as the best
          # k=beam_size scores are always included if all source_beam_sizes >= beam_size.
          if not source_beam_sizes:
            source_beam_sizes = [beam_size] * len(self.sources)
          assert len(source_beam_sizes) == len(self.sources), "Provide exactly one beam size per source."

          # Combine the incoming scores by adding them up for all possible combinations of target labels. To reduce
          # the number of combinations, we separately apply beam pruning to the sources beforehand.
          scores_in, scores_in_dim, pruned_labels = self._prune_and_combine_sources(
            self.sources, source_beam_sizes, net_batch_dim * beam_size)
          # scores_in has size (batch * beam_size, source_beam_sizes[0] * source_beam_sizes[1])
        else:
          scores_in = self._get_scores(self.sources[0])  # (batch * beam_size, dim)
          scores_in_dim = self.sources[0].output.dim
          if scores_in_dim is None:  # can happen if variable length
            scores_in_dim = tf.shape(self.sources[0].output.placeholder)[self.sources[0].output.feature_dim_axis]
          pruned_labels = None

        assert self.search_choices.src_layer, self.network.debug_search_choices(base_search_choice=self) or (
          "Not implemented yet. In rec-layer, we would always have our prev-frame as one previous search choice. "
          "Our deps: %r" % self.get_dep_layers())
        base_search_choices = self.search_choices.src_layer.search_choices
        scores_base = base_search_choices.beam_scores  # (batch, beam_in)
        assert scores_base.get_shape().ndims == 2, "%r invalid" % base_search_choices
        base_beam_in = tf.shape(scores_base)[1]  # 1 in first frame, then beam_in
        scores_beam_in = tf.shape(scores_in)[0] // net_batch_dim
        beam_in = self.sources[0].output.beam.beam_size
        assert beam_in == base_search_choices.beam_size, "%r: source %r beam-size unexpected from base choice %r" % (
          self, self.sources[0], base_search_choices)
        # About incoming beam size:
        #   base_beam_in  - 1 in first frame, then beam_in
        #   scores_beam_in  - beam_in or 1
        #   beam_in  - beam_in
        # Note about scores_beam_in, i.e. the batch-beam-size of other layers:
        # We could make it like base_beam_in, i.e. have beam-size 1 in the 0th layer
        # and also in the 1st layer before any ChoiceLayer.
        # However, currently it makes the code a bit simpler to just have always
        # the final beam-size everywhere.
        # Keep in mind that this might change at some future point.
        if self.length_normalization:
          assert self.network.have_rec_step_info()
          t = self.network.get_rec_step_index()  # scalar
          end_flags_flat = self.network.get_rec_step_info().get_prev_end_flag(
            target_search_choices=base_search_choices)  # (batch * beam_in,)
          with tf.name_scope("length_normalization"):
            end_flags = tf.reshape(end_flags_flat, [net_batch_dim, beam_in])  # (batch, beam_in)
            end_flags = end_flags[:, :base_beam_in]  # see scores_in below
            # Normalized scores, so we evaluate score_t/len.
            # If seq ended, score_t/t == score_{t-1}/(t-1), thus score_t = score_{t-1}*(t/(t-1))
            # Because we count with EOS symbol, shifted by one.
            scores_base *= tf.where(
              end_flags,
              tf.ones(tf.shape(end_flags)) * (
                tf.pow((tf.cast(t + 1, tf.float32) / tf.cast(t, tf.float32)), length_normalization_exponent)),
              tf.ones(tf.shape(end_flags)))
        scores_base = tf.expand_dims(scores_base, axis=-1)  # (batch, beam_in, dim)
        from returnn.tf.util.basic import filter_ended_scores
        if self.network.have_rec_step_info():
          scores_in = filter_ended_scores(
            scores_in,
            end_flags=self.network.get_rec_step_info().get_prev_end_flag(target_search_choices=base_search_choices),
            dim=scores_in_dim, batch_dim=net_batch_dim * scores_beam_in)  # (batch * beam_in, dim)
        scores_in = tf.reshape(scores_in, [net_batch_dim, scores_beam_in, scores_in_dim])  # (batch, beam_in, dim)
        with tf.control_dependencies([
              # See comment above. This checks that all is as expected.
              tf.Assert(tf.logical_or(
                tf.equal(base_beam_in, 1),
                tf.logical_and(
                  tf.equal(base_beam_in, scores_beam_in),
                  tf.equal(base_beam_in, beam_in))),
                [
                  "base_beam_in", base_beam_in,
                  "scores_beam_in", scores_beam_in,
                  "beam_in", beam_in])]):
          # See the comment above. It could be that scores_in has a wider beam
          # than what should be used here now.
          scores_in = scores_in[:, :base_beam_in]  # (batch, beam_in, dim)
        if custom_score_combine:
          with tf.name_scope("custom_score_combine"):
            if self.network.have_rec_step_info():
              t = self.network.get_rec_step_index()  # scalar
              end_flags_flat = self.network.get_rec_step_info().get_prev_end_flag(
                target_search_choices=base_search_choices)  # (batch * beam_in,)
              end_flags = tf.reshape(end_flags_flat, [net_batch_dim, beam_size])  # (batch, beam_in)
              end_flags = end_flags[:, :base_beam_in]  # see scores_in
            else:
              t, end_flags = None, None
            scores_comb = custom_score_combine(
              layer=self, scores_in=scores_in, scores_base=scores_base, t=t, end_flags=end_flags,
              batch_dim=net_batch_dim, scores_beam_in=scores_beam_in, base_beam_in=base_beam_in)
            assert isinstance(scores_comb, tf.Tensor)
        else:
          scores_random_sample = None
          if random_sample_scale:
            # https://github.com/tensorflow/tensorflow/issues/9260
            # https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
            scores_random_sample = -tf_compat.v1.log(-tf_compat.v1.log(
              tf_compat.v1.random_uniform(tf.shape(scores_in), 0, 1)))
          scores_comb = optional_add(
            optional_mul(scores_in, prob_scale),
            optional_mul(scores_base, base_beam_score_scale),
            optional_mul(scores_random_sample, random_sample_scale))  # (batch, beam_in, dim)
        scores_comb.set_shape(
          (None, None, None if isinstance(scores_in_dim, tf.Tensor) else scores_in_dim))  # (batch, beam_in, dim)
        self.search_scores_in = scores_in
        self.search_scores_base = scores_base
        self.search_scores_combined = scores_comb
        cheating_gold_targets, cheating_src_beam_idx = None, None
        cheating_exclusive = False
        if cheating:
          cheating_gold_targets, cheating_src_beam_idx = self._get_cheating_targets_and_src_beam_idxs(scores_comb)
          if isinstance(cheating, bool):
            pass
          elif isinstance(cheating, str):
            assert cheating == "exclusive", "%s: invalid cheating %r" % (self, cheating)  # only possible variation atm
            cheating_exclusive = True
          else:
            raise TypeError("%s: invalid cheating %r" % (self, cheating))
        # `tf.nn.top_k` is the core function performing our search.
        # That is wrapped in `returnn.tf.util.basic.beam_search`.
        # We get scores/labels of shape (batch, beam) with indices in [0..beam_in*dim-1].
        from returnn.tf.util.basic import beam_search
        src_beams, labels, scores = beam_search(
          scores=scores_comb, beam_size=beam_size, keep_beams=keep_beams,
          cheating_gold_targets=cheating_gold_targets, cheating_src_beam_idx=cheating_src_beam_idx,
          cheating_exclusive=cheating_exclusive)
        self.search_choices.set_src_beams(src_beams)  # (batch, beam) -> beam_in idx
        labels = tf.reshape(labels, [net_batch_dim * beam_size])  # (batch * beam)
        labels = tf.cast(labels, self.output.dtype)

        if len(self.sources) > 1:
          # 'labels' in this case do not refer to a target vocabulary, but just represent ids to the labels
          # that survived pruning for each of the sources ('pruned_labels'). So as a last step, we get the final
          # target labels by indexing pruned_labels with 'labels'.
          labels = self._get_combined_labels(labels, src_beams, pruned_labels, source_beam_sizes)
        else:
          labels = [labels]

        self.search_choices.set_beam_scores(scores)  # (batch, beam) -> log score
        if self._debug_out is not None:
          from returnn.tf.util.basic import identity_with_debug_log
          labels[0] = identity_with_debug_log(
            out=self._debug_out, x=labels[0], args={
              "step": self.network.get_rec_step_index() if self.network.have_rec_step_info() else tf.constant(-1),
              "base_beam_in": base_beam_in,
              "scores_in_orig": self.sources[0].output.placeholder,
              "scores_in": scores_in,
              "scores_base_orig": self.search_choices.src_layer.search_choices.beam_scores,
              "scores_base": scores_base,
              "scores_combined": scores_comb,
              "src_beam_idxs": self.search_choices.src_beams,
              "labels": tf.reshape(labels[0], [net_batch_dim, beam_size]),
              "scores": scores})

        # Put labels for all targets in a list.
        # They can be accessed by using the sublayers created in self.get_sub_layer().
        self.output_list = []
        for index, labels_ in enumerate(labels):
          self.output_list.append(Data(
            name="%s_choice_output_%d" % (self.name, index),
            batch_dim_axis=0,
            dim_tags=self.output.dim_tags,
            sparse=True,
            sparse_dim=self.sources[index].output.feature_dim_or_sparse_dim,
            dtype=self.output.dtype,
            placeholder=labels_,
            available_for_inference=True,
            batch=self.output.batch,
            beam=self.output.beam))

        # We use the labels of the first target as "normal" output.
        self.output = self.output_list[0]

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
      if input_type == "regression":
        feedback_output = self.sources[0].output.get_placeholder_as_batch_major()  # (batch, dim)
      else:
        # sample from scores
        scores_in = self._get_scores(self.sources[0])  # +log scores, (batch, dim)
        feedback_output = tf_compat.v1.multinomial(
          scores_in, num_samples=1, seed=tf_util.get_random_seed())  # (batch, num_samples), int64
        feedback_output = tf.cast(tf.reshape(feedback_output, [-1]), tf.int32)  # (batch,), int32

      gold_mixing_prob = self.scheduled_sampling.get("gold_mixin_prob", False)
      if gold_mixing_prob:
        gold_targets = self._get_target_value().get_placeholder_as_batch_major()
        # draw choices over batch dimension
        choice = tf.less(tf_compat.v1.random_uniform(tf.shape(feedback_output)[:1]), gold_mixing_prob)
        feedback_output = tf.where(choice, gold_targets, feedback_output)

      self.output = Data(
        name="%s_sampled_output" % self.name,
        batch_dim_axis=0,
        dim_tags=self.output.dim_tags,
        batch=self.output.batch,
        sparse=input_type != "regression",
        dim=self.output.dim,
        dtype=self.output.dtype,
        placeholder=feedback_output,
        available_for_inference=True)

    else:  # no search, and no scheduled-sampling
      # Note: If you want to do forwarding, without having the reference,
      # that wont work. You must do search in that case.
      # Put all targets in a list.
      # They can be accessed by using the sublayers created in self.get_sub_layer().
      self.output_list = []
      for target in self.targets:
        target_out_data = self._static_get_target_value(
          target=target, network=self.network, mark_data_key_as_used=True,
          search_choices=self.get_search_choices()).copy()
        target_out_data.available_for_inference = True  # in inference, we should do search
        assert target_out_data.placeholder is not None
        self.output_list.append(target_out_data)

      # We use the labels of the first target as "normal" output.
      self.output = self.output_list[0]

    if add_to_beam_scores and self.sources[0].output.beam and not search and input_type != "regression":
      # We perform search, but this layer does not do search.
      # But we still add our scores to the beam scores.
      net_batch_dim = self.network.get_data_batch_dim()
      base_search_choices = self.network.get_search_choices(base_search_choice=self).search_choices
      self.search_choices = SearchChoices(
        owner=self,
        beam_size=base_search_choices.beam_size)
      assert self.search_choices.beam_size == self.output.beam.beam_size
      scores_base = base_search_choices.beam_scores  # (batch, beam_in|1)
      assert len(self.sources) == 1
      scores_in = self._get_scores(self.sources[0])  # +log scores, (batch*beam_in, dim)
      from returnn.tf.util.basic import filter_ended_scores
      if self.network.have_rec_step_info():
        scores_in_dim = self.sources[0].output.dim
        if scores_in_dim is None:  # can happen if variable length
          scores_in_dim = tf.shape(self.sources[0].output.placeholder)[self.sources[0].output.feature_dim_axis]
        scores_in = filter_ended_scores(
          scores_in,
          end_flags=self.network.get_rec_step_info().get_prev_end_flag(target_search_choices=base_search_choices),
          dim=scores_in_dim, batch_dim=tf.shape(scores_in)[0])  # (batch * beam_in, dim)
        # We also assume that the ground truth output are 0 when the seq ended.
      scores_in_ = batch_gather(scores_in, self.output.placeholder)  # (batch*beam_in,)
      scores_in_ = tf.reshape(scores_in_, (net_batch_dim, base_search_choices.beam_size))  # (batch,beam_in)
      self.search_choices.set_src_beams(expand_dims_unbroadcast(
        tf.range(base_search_choices.beam_size), axis=0, dim=net_batch_dim))
      assert not random_sample_scale
      assert not length_normalization
      assert not custom_score_combine
      scores_comb = optional_add(
        optional_mul(scores_in_, prob_scale),
        optional_mul(scores_base, base_beam_score_scale))  # (batch, beam_in)
      self.search_scores_in = scores_in_
      self.search_scores_base = scores_base
      self.search_scores_combined = scores_comb
      self.search_choices.set_beam_scores(scores_comb)

  def _get_scores(self, source):
    """
    :param LayerBase source:
    :return: scores in +log space, (batch,feature), batch might include beam
    :rtype: tf.Tensor
    """
    assert source.output.is_batch_major
    scores_in = source.output.placeholder
    # We present the scores in +log space, and we will add them up along the path.
    if self.input_type == "prob":
      if source.output_before_activation:
        return source.output_before_activation.get_log_output()
      else:
        from returnn.tf.util.basic import safe_log
        return safe_log(scores_in)
    elif self.input_type == "logits":
      return tf.nn.log_softmax(scores_in)
    elif self.input_type == "log_prob":
      return scores_in
    else:
      raise Exception("%r: invalid input type %r" % (self, self.input_type))

  def _prune_and_combine_sources(self, sources, beam_sizes, batch_dim):
    """
    Applies beam pruning to the sources and then calculates all possible sums of scores.
    Returns the scores, the (static) number of targets after pruning and a list of
    labels corresponding to the top scores.

    :param list[LayerBase] sources: input layers providing the scores
    :param list[int] beam_sizes: beam sizes used for pruning of the individual sources
    :param tf.Tensor|int batch_dim: dim of batch axis (batch size * incoming beam)
    :return: combined scores, dim of combined scores, labels that survived pruning
    :rtype: (tf.Tensor, int, list[tf.Tensor])
    """
    # Calculate the product of beam_sizes. This will be the length (i.e. 'dim') of combined_pruned_scores.
    combined_scores_dim = 1

    pruned_scores = []
    pruned_labels = []

    with tf.name_scope("combine_sources"):
      # prune incoming sources separately
      for source, beam_size in zip(sources, beam_sizes):
        scores_in = self._get_scores(source)

        scores, labels = tf.nn.top_k(scores_in, k=beam_size)
        pruned_scores.append(scores)
        pruned_labels.append(labels)

        combined_scores_dim *= beam_size

      # We want to compute scores for all possible combination of sources. This is done by putting each source
      # on a separate axis. This leads to broadcasting of all source-axes over all others when adding up the sources,
      # thus giving all possible sums of scores. We reshape back afterwards.
      num_sources = len(sources)
      combined_pruned_scores = None

      for source_index, pruned_scores_this_source in enumerate(pruned_scores):
        expanded_pruned_scores = pruned_scores_this_source

        # Put n-th source on the axis n+1 (first axis is the batch dim).
        for axis_index in range(num_sources):
          if axis_index != source_index:
            expanded_pruned_scores = tf.expand_dims(expanded_pruned_scores, axis_index + 1)  # +1 because of batch dim

        # Accumulatively add up to previous scores.
        if combined_pruned_scores is None:
          combined_pruned_scores = expanded_pruned_scores
        else:
          combined_pruned_scores += expanded_pruned_scores

      # We flatten over the beam dims of the sources, but not yet over the batch dim. This matches
      # the shape of the input scores in case of a single source.
      combined_pruned_scores_flat = tf.reshape(combined_pruned_scores, [batch_dim, combined_scores_dim])

    return combined_pruned_scores_flat, combined_scores_dim, pruned_labels

  # noinspection PyMethodMayBeStatic
  def _get_combined_labels(self, combined_ids, src_beams, pruned_labels, beam_sizes):
    """
    Gets output labels by converting 'combined_ids' (corresponding to the flattend shape created in
    self._prune_and_combine_sources()) back to separate ids and then using those as indices to the labels
    that survived pruning.

    :param tf.Tensor combined_ids: indices to the flattened scores, see self._prune_and_combine_sources()
    :param tf.Tensor src_beams: the indices of the incoming beam for each outgoing label
    :param list[tf.Tensor] pruned_labels: labels after pruning of incoming beam, see self._prune_and_combine_sources()
    :param list[int] beam_sizes: beam sizes used for pruning of the individual sources
    :return: final labels for all sources
    :rtype: list[tf.Tensor]
    """
    from returnn.tf.util.basic import batch_gather
    with tf.name_scope("get_combined_labels"):
      # For each target we first have to get the labels that survived source pruning from the beam index
      # the outgoing label was generated from. So choose 'pruned_labels' according to 'src_beams'.
      pruned_labels_src_beam_selected = []
      for index, pruned_labels_ in enumerate(pruned_labels):
        pruned_labels_src_beam_selected.append(tf_util.select_src_beams(pruned_labels_, src_beams))

      # We can recover the ids for the unflattened shape by using integer division and modulo operations.
      # (similar to numpy.unravel_index())
      ids = []
      # reversed because innermost dim, which is unflattened first, corresponds to last target
      for beam_size in reversed(beam_sizes):
        ids_ = combined_ids % beam_size
        combined_ids //= beam_size
        ids.append(ids_)

      ids = reversed(ids)  # because we created it backwards

      # Now get the final target labels by indexing the incoming labels that survived pruning.
      labels = []
      for pruned_labels_src_beam_selected_, ids_ in zip(pruned_labels_src_beam_selected, ids):
        labels_ = tf.squeeze(batch_gather(pruned_labels_src_beam_selected_, tf.expand_dims(ids_, axis=-1)), axis=-1)
        labels.append(labels_)

      return labels

  def _get_cheating_targets_and_src_beam_idxs(self, scores):
    """
    :param tf.Tensor scores: (batch,beam_in,dim). combined scores (i.e. base beam scores + new scores),
      dense over the dims, such that we have labels in [0,...,dim-1].
    :return: cheating_gold_targets, cheating_src_beam_idx
    :rtype: (tf.Tensor,tf.Tensor|None)
    """
    assert self.cheating
    assert len(self.sources) == 1, "Cheating not yet implemented for multiple sources."
    cheating_gold_targets = self._get_target_value(
      search_choices=None).get_placeholder_as_batch_major()  # (batch,), int32
    base_search_choices = self.search_choices.src_layer.search_choices
    assert isinstance(base_search_choices, SearchChoices)
    other_choice_layer = base_search_choices.owner.get_normalized_layer()
    if other_choice_layer is self:  # self from prev frame
      return cheating_gold_targets, None  # default case for returnn.tf.util.basic.beam_search
    # Different choice.
    if not isinstance(other_choice_layer, ChoiceLayer):
      # Warning: This is wrong in general. (It might be correct depending on your config.)
      # However, this is the old behavior.
      return cheating_gold_targets, None  # default case for returnn.tf.util.basic.beam_search
    assert isinstance(other_choice_layer, ChoiceLayer)  # else not implemented
    if other_choice_layer.cheating:  # also cheating?
      return cheating_gold_targets, None  # default case for returnn.tf.util.basic.beam_search
    # We must know valid cheating_src_beam_idx which are from cheating traces.
    other_choice_src_layer = other_choice_layer.search_choices.src_layer
    # Note: We cannot used get_normalized_layer because `self` is not registered yet.
    assert other_choice_src_layer.network is self.network and other_choice_src_layer.name == "prev:%s" % self.name, (
      other_choice_layer, other_choice_layer.search_choices, other_choice_src_layer,
      "Expected that this is self from prev frame.")
    # Figure out the beam index of ourselves from the previous frame.
    # Normally this is beam_in - 1. beam_in is the incoming beam_size, which is 1 in the first frame.
    # Also be careful about end-of-sequence.
    # A somewhat simple way to determine it for all these cases:
    prev_beam_idx = tf.reduce_max(base_search_choices.src_beams)
    # Now find the best possible beam index.
    # Note that we could do this even in the general case.
    # It also would make sense later to not add the cheating label twice (see returnn.tf.util.basic.beam_search logic);
    # in that case, we always must use this logic here.
    from returnn.tf.util.basic import get_shape, where_bc, beam_search
    n_batch, beam_in, dim = get_shape(scores)
    scores = where_bc(
      tf.expand_dims(tf.equal(base_search_choices.src_beams, prev_beam_idx), axis=-1),
      scores,
      float("-inf"))  # (batch,beam_in,dim)
    scores = where_bc(
      tf.equal(cheating_gold_targets[:, None, None], tf.range(dim)[None, None, :]),
      scores,
      float("-inf"))  # (batch,beam_in,dim)
    src_beams, _, _ = beam_search(scores, beam_size=1)
    src_beams = src_beams[:, 0]  # (batch,)
    return cheating_gold_targets, src_beams

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    assert d.get("from", NotSpecified) is not NotSpecified, "specify 'from' explicitly for choice layer"
    if not isinstance(d["from"], (tuple, list)):
      d["from"] = [d["from"]]
    search = d.get("search", NotSpecified)
    search = NotSpecified.resolve(search, network.search_flag)
    add_to_beam_scores = d.get("add_to_beam_scores", NotSpecified)
    add_to_beam_scores = NotSpecified.resolve(add_to_beam_scores, search or network.search_flag)
    if d.get("target", NotSpecified) is not None:
      assert "target" in d, "%s: specify 'target' explicitly" % (cls.__name__,)
      if isinstance(d["target"], str):
        d["target"] = [d["target"]]
      assert isinstance(d["target"], list)
      assert len(d["target"]) == len(d["from"])
    if not search and not add_to_beam_scores and not d.get("scheduled_sampling"):
      # In the dependency graph, we don't want it.
      # This can enable some optimizations in the RecLayer.
      # We do it here because we should know about the deps early in the template creation in RecLayer.
      # Note that we don't look at d.get("search") here, because in case of search,
      # if there are other choice layers, we still need to add the scores to the beam.
      d["from"] = []

    # Convert explicit_search_sources to layers
    if d.get("explicit_search_source"):
      assert "explicit_search_sources" not in d
      d["explicit_search_sources"] = [get_layer(d.pop("explicit_search_source"))] if search else []
    elif d.get("explicit_search_sources"):
      assert isinstance(d["explicit_search_sources"], (list, tuple, dict))
      if search:
        if isinstance(d["explicit_search_sources"], (list, tuple)):
          d["explicit_search_sources"] = [get_layer(name) for name in d["explicit_search_sources"]]
        elif isinstance(d["explicit_search_sources"], dict):
          d["explicit_search_sources"] = {key: get_layer(name) for key, name in d["explicit_search_sources"].items()}
      else:
        d["explicit_search_sources"] = []
    super(ChoiceLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

  @classmethod
  def _create_search_beam(cls, name, beam_size, sources, network):
    """
    :param str name:
    :param int beam_size:
    :param list[LayerBase] sources:
    :param returnn.tf.network.TFNetwork network:
    :rtype: returnn.tf.util.data.SearchBeam
    """
    from returnn.tf.util.basic import SearchBeam
    search_dep = sources[0].output.beam
    return SearchBeam(
      beam_size=beam_size, dependency=search_dep,
      name="%s%s" % (network.get_absolute_name_prefix(), name))

  @classmethod
  def get_out_data_from_opts(cls, name, sources, target, network,
                             beam_size, search=NotSpecified, scheduled_sampling=False, cheating=False, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str target:
    :param returnn.tf.network.TFNetwork network:
    :param int beam_size:
    :param NotSpecified|bool search:
    :param dict|bool scheduled_sampling:
    :param bool cheating:
    :rtype: Data
    """
    search = NotSpecified.resolve(search, network.search_flag)
    target = target[0] if isinstance(target, list) else target  # only the first matters here
    if target:
      out_data = cls._static_get_target_value(
        target=target, network=network, mark_data_key_as_used=False).copy_template(name="%s_output" % name)
      out_data.available_for_inference = True  # in inference, we would do search
    else:  # no target. i.e. we must do search
      assert search, "%s %r: no target given, must do search" % (cls.__name__, name)
      # Output will be the sparse version of the input.
      out_data = sources[0].output.copy_template().copy_as_batch_major()
      dim_tags = list(out_data.dim_tags)
      feature_dim = dim_tags.pop(out_data.feature_dim_axis)
      out_data = Data(
        name="%s_output" % name, dim_tags=dim_tags, sparse=True, dim=out_data.dim, sparse_dim=feature_dim,
        batch=out_data.batch.copy_set_beam(None) if out_data.batch else network.get_global_batch_info())
    if search:
      out_data.beam = cls._create_search_beam(name=name, beam_size=beam_size, sources=sources, network=network)
    elif sources and sources[0]:
      out_data.beam = sources[0].output.beam
    if out_data.beam and out_data.batch:
      out_data.batch = out_data.batch.copy_set_beam(out_data.beam)
    if cheating or scheduled_sampling or not search:
      cls._static_get_target_value(target=target, network=network, mark_data_key_as_used=True)  # mark as used
    return out_data

  def get_sub_layer(self, layer_name):
    """
    Used to get outputs in case of multiple targets. For all targets we create a sub-layer that can be referred to
    as "self.name + '/out_' + index" (e.g. output/out_0). These sub-layers can then be used as input to other layers,
    e.g. "output_0": {"class": "copy", "from": ["output/out_0"].

    :param str layer_name: name of the sub_layer (e.g. 'out_0')
    :return: internal layer that outputs labels for the target corresponding to layer_name
    :rtype: InternalLayer
    """
    from .base import InternalLayer
    assert layer_name.startswith("out_")
    index = int(layer_name[len("out_"):])
    sub_layer = InternalLayer(
      name="%s/%s" % (self.name, layer_name), network=self.network, output=self.output_list[index], sources=[self])
    return sub_layer

  @classmethod
  def get_sub_layer_out_data_from_opts(cls, layer_name, parent_layer_kwargs):
    """
    :param str layer_name: name of the sub_layer (e.g. 'out_0'), see self.get_sub_layer()
    :param dict[str] parent_layer_kwargs: kwargs for the parent layer
    :return: Data template, class type of sub-layer, layer opts (transformed)
    :rtype: (Data, type, dict[str])|None
    """
    assert layer_name.startswith("out_")
    index = int(layer_name[len("out_"):])

    targets = parent_layer_kwargs["target"]
    assert isinstance(targets, list), "Sub-layers for ChoiceLayer should only exist in case of multiple targets."

    parent_layer_name = parent_layer_kwargs["name"]
    sources = parent_layer_kwargs["sources"]
    network = parent_layer_kwargs["network"]
    beam_size = parent_layer_kwargs["beam_size"]
    search = parent_layer_kwargs.get("search", NotSpecified)
    assert isinstance(network, TFNetwork)
    search = NotSpecified.resolve(search, network.search_flag)

    # The sub-layer with index n will output the n-th target. The out_data is taken directly
    # from the target as it is done in self.get_out_data_from_opts().
    sub_layer_out_data = cls.get_out_data_from_opts(
      name=parent_layer_name + "/" + layer_name,
      sources=sources, target=targets[index], network=network, beam_size=beam_size)

    if search:
      # Create same beam as in parent layer (they have to compare equal)
      sub_layer_out_data.beam = cls._create_search_beam(
        name=parent_layer_name, beam_size=beam_size,
        sources=sources, network=network)

    from .base import InternalLayer
    return sub_layer_out_data, InternalLayer, {}

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    # See also self.transform_config_dict where we might strip away the sources.
    ls = super(ChoiceLayer, self).get_dep_layers()
    if self.explicit_search_sources:
      if isinstance(self.explicit_search_sources, dict):
        additional_sources = [src for _, src in sorted(self.explicit_search_sources.items())]
      else:
        additional_sources = self.explicit_search_sources
      ls.extend(additional_sources)
    return ls

  def post_process_final_rec_vars_outputs(self, rec_vars_outputs, seq_len):
    """
    :param dict[str,tf.Tensor] rec_vars_outputs:
    :param tf.Tensor seq_len: shape (batch,)
    :rtype: dict[str,tf.Tensor]
    """
    if self.length_normalization:
      assert "choice_scores" in rec_vars_outputs
      rec_layer = self.network.get_rec_parent_layer()
      assert rec_layer
      # Finalize length normalization. During search we keep an extra factor t (recurrent time step) for efficiency
      # reasons (see self.get_output()). Remove it here.
      num_time_steps = tf.reduce_max(seq_len)
      if not rec_layer.include_eos:
        num_time_steps += 1  # + 1 to include sequence end
      rec_vars_outputs["choice_scores"] /= tf.cast(num_time_steps, tf.float32)
    return rec_vars_outputs


class DecideLayer(BaseChoiceLayer):
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
    super(DecideLayer, self).__init__(beam_size=1, **kwargs)
    assert len(self.sources) == 1
    src = self.sources[0]
    if src.output.beam:
      self.output, self.search_choices = self.decide(
        src=src, owner=self, output=self.output, length_normalization=length_normalization)
      assert self.search_choices
    else:
      if self.network.search_flag:
        print("%s: Warning: decide on %r, there are no search choices" % (self, src), file=log.v3)
      self.output.placeholder = src.output.copy_as_batch_major().placeholder

  @classmethod
  def cls_get_search_beam_size(cls, sources, **kwargs):
    """
    :param list[LayerBase] sources:
    :rtype: int|None
    """
    if sources[0].output.beam:
      return 1
    return None

  @classmethod
  def decide(cls, src, output=None, owner=None, name=None, length_normalization=False):
    """
    :param LayerBase src: with search_choices set. e.g. input of shape (batch * beam, time, dim)
    :param Data|None output:
    :param LayerBase|None owner:
    :param str|None name:
    :param bool length_normalization: performed on the beam scores
    :return: best beam selected from input, e.g. shape (batch, time, dim)
    :rtype: (Data, SearchChoices|None)
    """
    search_choices = src.get_search_choices()
    if not search_choices:
      return src.output, None
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
      beam_scores /= tf.cast(tf.reshape(src.output.get_sequence_lengths(), [batch_dim, beam_size]), tf.float32)
    beam_idxs = tf.argmax(beam_scores, axis=1)  # (batch,)
    from returnn.tf.util.basic import assert_min_tf_version, nd_indices, Dim
    assert_min_tf_version((1, 1), "gather_nd")
    beam_idxs_ext = nd_indices(beam_idxs)
    output.placeholder = tf.cond(
      tf.greater(tf.size(src_output), 0),  # can happen to be empty
      lambda: tf.gather_nd(src_output, indices=beam_idxs_ext),
      lambda: src_output[:, 0], name="cond_not_empty")  # (batch, [time], [dim])
    output.size_placeholder = {}
    for i, size in src_data.size_placeholder.items():
      tag = Dim.get_tag_from_size_tensor(size)
      assert tag
      tag = tag.get_for_batch_ctx(batch=output.batch, ctx=output.control_flow_ctx)
      if tag.dyn_size is None:
        size = tf.reshape(size, [batch_dim, beam_size])  # (batch, beam)
        size = tf.gather_nd(size, indices=beam_idxs_ext)  # (batch,)
        tag.set_tag_on_size_tensor(size, batch=output.batch)
      else:
        size = tag.dyn_size
      output.size_placeholder[i] = size
      # The max(size) might have become smaller than before, so we should slice the output
      # such that output.shape[i] == max(size).
      slices = [slice(None, None)] * src_data.get_batch_axis(i) + [slice(None, tf.reduce_max(size))]
      output.placeholder = output.placeholder[slices]
    final_search_choices = SearchChoices(owner=owner, is_decided=True, beam_size=1)
    if owner:
      final_search_choices.set_src_beams(tf.expand_dims(beam_idxs, axis=1))
      final_search_choices.set_beam_scores(tf.expand_dims(tf.gather_nd(beam_scores, indices=beam_idxs_ext), axis=1))
    return output, final_search_choices

  @classmethod
  def get_out_data_from_opts(cls, name, sources, network, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param returnn.tf.network.TFNetwork network:
    :rtype: Data
    """
    assert len(sources) == 1
    data = sources[0].output.copy_template(name="%s_output" % name).copy_as_batch_major()
    data.beam = None
    return data


class DecideKeepBeamLayer(BaseChoiceLayer):
  """
  This just marks the search choices as decided, but does not change them (in contrast to :class:`DecideLayer`).
  You can use this to get out some values as-is, without having them resolved to the final choices.

  For internal usage only.
  """
  layer_class = "decide_keep_beam"

  def __init__(self, sources, **kwargs):
    """
    :param list[LayerBase] sources:
    """
    assert len(sources) == 1
    src = sources[0]
    # We also allow a source after a DecideLayer.
    beam_size = src.output.beam.beam_size if src.output.beam else 1
    super(DecideKeepBeamLayer, self).__init__(beam_size=beam_size, sources=sources, **kwargs)
    self.output.placeholder = src.output.placeholder
    base_search_choices = src.get_search_choices()
    if base_search_choices:
      self.search_choices = SearchChoices(owner=self, beam_size=beam_size, keep_raw=True)
      assert base_search_choices.beam_size == beam_size == self.search_choices.beam_size
      net_batch_dim = self.network.get_data_batch_dim()
      from returnn.tf.util.basic import expand_dims_unbroadcast
      self.search_choices.set_src_beams(expand_dims_unbroadcast(
        tf.range(base_search_choices.beam_size), axis=0, dim=net_batch_dim))
      self.search_choices.set_beam_scores(base_search_choices.beam_scores)
    else:
      if self.network.search_flag:
        print("%s: Warning: decide-keep-beam on %r, there are no search choices" % (self, src), file=log.v3)

  @classmethod
  def cls_get_search_beam_size(cls, sources, network, **kwargs):
    """
    :param list[LayerBase] sources:
    :param returnn.tf.network.TFNetwork network:
    :rtype: int|None
    """
    assert len(sources) == 1
    return sources[0].output.beam.beam_size if sources[0].output.beam else None

  @classmethod
  def get_rec_initial_extra_outputs(cls, sources, **kwargs):
    """
    :param list[LayerBase] sources:
    :rtype: dict[str,tf.Tensor]
    """
    assert len(sources) == 1
    return super(DecideKeepBeamLayer, cls).get_rec_initial_extra_outputs(
      beam_size=sources[0].output.beam.beam_size, sources=sources, **kwargs)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    d.setdefault("from", [])  # using "data" does not make much sense
    d.setdefault("collocate_with", d["from"])  # should be right where the source is
    super(DecideKeepBeamLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

  @classmethod
  def get_out_data_from_opts(cls, name, sources, network, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param returnn.tf.network.TFNetwork network:
    :rtype: Data
    """
    from returnn.tf.util.basic import SearchBeam
    assert len(sources) == 1
    out = sources[0].output.copy_template(name="%s_output" % name)
    if out.beam:
      out.beam = SearchBeam(
        beam_size=out.beam.beam_size, dependency=out.beam,
        name="%s%s" % (network.get_absolute_name_prefix(), name))
    return out


class ChoiceGetBeamScoresLayer(LayerBase):
  """
  Gets beam scores from :class:`SearchChoices`.
  This requires that the source has search choices.

  .. note::

    This layer might be deprecated in the future.

  """
  layer_class = "choice_get_beam_scores"

  def __init__(self, **kwargs):
    super(ChoiceGetBeamScoresLayer, self).__init__(**kwargs)
    assert len(self.sources) == 1
    search_choices = self.sources[0].get_search_choices()
    assert search_choices, "%s: source %s has no search choices" % (self, self.sources[0])
    assert search_choices.beam_size == self.output.beam.beam_size
    net_batch_dim = self.network.get_data_batch_dim()
    self.output.placeholder = tf.reshape(search_choices.beam_scores, [net_batch_dim * search_choices.beam_size])

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    d.setdefault("from", [])  # using "data" does not make much sense
    d.setdefault("collocate_with", d["from"])  # should be right where the source is
    super(ChoiceGetBeamScoresLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

  @classmethod
  def get_out_data_from_opts(cls, name, sources, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    assert len(sources) == 1
    return Data(name="%s_output" % name, dtype="float32", shape=(), beam=sources[0].output.beam)


class ChoiceGetSrcBeamsLayer(LayerBase):
  """
  Gets source beam indices from :class:`SearchChoices`.
  This requires that the source has search choices.
  """
  layer_class = "choice_get_src_beams"

  def __init__(self, **kwargs):
    super(ChoiceGetSrcBeamsLayer, self).__init__(**kwargs)
    assert len(self.sources) == 1
    search_choices = self.sources[0].get_search_choices()
    assert search_choices, "%s: source %s has no search choices" % (self, self.sources[0])
    assert search_choices.beam_size == self.output.beam.beam_size
    net_batch_dim = self.network.get_data_batch_dim()
    self.output.placeholder = tf.reshape(search_choices.src_beams, [net_batch_dim * search_choices.beam_size])

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    d.setdefault("from", [])  # using "data" does not make much sense
    d.setdefault("collocate_with", d["from"])  # should be right where the source is
    super(ChoiceGetSrcBeamsLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

  @classmethod
  def get_out_data_from_opts(cls, name, sources, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    assert len(sources) == 1
    return Data(name="%s_output" % name, dtype="int32", shape=(), dim=None, beam=sources[0].output.beam)


class SplitBatchBeamLayer(BaseChoiceLayer):
  """
  Splits the batch dimension of the input, which includes a beam, into (batch,beam).

  Like :class:`DecideLayer`, this removes the beam.
  """
  layer_class = "split_batch_beam"

  def __init__(self, beam_dim=None, **kwargs):
    """
    :param Dim|None beam_dim:
    """
    beam_dim  # noqa  # via get_out_data_from_opts
    super(SplitBatchBeamLayer, self).__init__(beam_size=1, **kwargs)
    src = self.sources[0].output.copy_as_batch_major()
    assert src.beam
    batch_dim = self.output.get_batch_dim()
    beam_size = src.beam.beam_size
    src_shape = tf_util.get_shape(src.placeholder)
    self.output.placeholder = tf.reshape(src.placeholder, [batch_dim, beam_size] + src_shape[1:])
    self.search_choices = SearchChoices(owner=self, beam_size=1, is_decided=True)

  @classmethod
  def cls_get_search_beam_size(cls, **kwargs):
    """
    :rtype: int|None
    """
    # We (currently) assume in this layer that there is always a beam.
    return 1

  @classmethod
  def get_out_data_from_opts(cls, name, network, sources, beam_dim=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param returnn.tf.network.TFNetwork network:
    :param Dim|None beam_dim:
    :rtype: Data
    """
    assert len(sources) == 1
    data = sources[0].output.copy_template(name="%s_output" % name).copy_as_batch_major()
    beam = data.beam
    assert beam, "no beam in %r" % data
    data.beam = None
    if beam_dim is None:
      beam_dim = SpatialDim("beam:%s" % beam.name, beam.beam_size, auto_generated=True)
    assert beam_dim.dimension == beam.beam_size
    data = data.copy_add_dim_by_tag(beam_dim, unbroadcast=True, axis=1)
    return data


class AttentionBaseLayer(_ConcatInputLayer):
  """
  This is the base class for attention.
  This layer would get constructed in the context of one single decoder step.
  We get the whole encoder output over all encoder frames (the base), e.g. (batch,enc_time,enc_dim),
  and some current decoder context, e.g. (batch,dec_att_dim),
  and we are supposed to return the attention output, e.g. (batch,att_dim).

  Some sources:
  * Bahdanau, Bengio, Montreal, Neural Machine Translation by Jointly Learning to Align and Translate, 2015,
    https://arxiv.org/abs/1409.0473
  * Luong, Stanford, Effective Approaches to Attention-based Neural Machine Translation, 2015,
    https://arxiv.org/abs/1508.04025
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
    self.base_weights = None  # type: typing.Optional[tf.Tensor]  # (batch, base_time), see self.get_base_weights()

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
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
    from returnn.tf.util.basic import assert_min_tf_version, nd_indices
    assert_min_tf_version((1, 1), "gather_nd")
    last_frame_idxs_ext = nd_indices(last_frame_idxs)
    return tf.gather_nd(self.get_base_weights(), indices=last_frame_idxs_ext)  # (batch,)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    super(AttentionBaseLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["base"] = get_layer(d["base"])

  @classmethod
  def get_out_data_from_opts(cls, name, base, n_out=NotSpecified, sources=(), **kwargs):
    """
    :param str name:
    :param int|None|NotSpecified n_out:
    :param LayerBase base:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    out = base.output.copy_template_excluding_time_dim().copy(name="%s_output" % name)
    assert out.batch_dim_axis == 0
    if n_out is not NotSpecified:
      assert out.dim == n_out, (
        "The default attention selects some frame-weighted input of shape [batch, frame, dim=%i]," % out.dim +
        " thus resulting in [batch, dim=%i] but you specified n_out=%i." % (out.dim, n_out))
    out.beam = SearchBeam.get_combined_beam(out.beam, *[src.output.beam for src in sources if src])
    return out


class GlobalAttentionContextBaseLayer(AttentionBaseLayer):
  """
  Base class for other attention types, which use a global context.
  """

  def __init__(self, base_ctx, **kwargs):
    """
    :param LayerBase base: encoder output to attend on
    :param LayerBase base_ctx: encoder output used to calculate the attention weights
    """
    super(GlobalAttentionContextBaseLayer, self).__init__(**kwargs)
    self.base_ctx = base_ctx

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    return super(GlobalAttentionContextBaseLayer, self).get_dep_layers() + [self.base_ctx]

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    super(GlobalAttentionContextBaseLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["base_ctx"] = get_layer(d["base_ctx"])


class GenericAttentionLayer(AttentionBaseLayer):
  """
  The weighting for the base is specified explicitly here.
  This can e.g. be used together with :class:`SoftmaxOverSpatialLayer`.
  Note that we do not do any masking here. E.g. :class:`SoftmaxOverSpatialLayer` does that.

  Note that :class:`DotLayer` is similar, just using a different terminology.
  Reduce axis: weights: time-axis; base: time-axis.
    Note that if the last layer was :class:`SoftmaxOverSpatialLayer`, we should use the same time-axis.
    Also we should do a check whether these time axes really match.
  Common axes (should match): batch-axis, all from base excluding base feature axis and excluding time axis.
  Keep axes: base: feature axis; weights: all remaining, e.g. extra time.
  """
  layer_class = "generic_attention"
  recurrent = True  # operates over spatial axis

  def __init__(self, weights, auto_squeeze=True, **kwargs):
    """
    :param LayerBase base: encoder output to attend on. (B, enc-time)|(enc-time, B) + (...) + (n_out,)
    :param LayerBase weights: attention weights. ((B, enc-time)|(enc-time, B)) + (1,)|()
    :param bool auto_squeeze: auto-squeeze any weight-axes with dim=1 away
    """
    super(GenericAttentionLayer, self).__init__(**kwargs)
    self.weights = weights
    assert not self.sources, "only base and weights are needed"

    from .basic import DotLayer, InternalLayer
    if not weights.output.is_batch_major:
      weights = InternalLayer(
        network=weights.network, name="%s_batch_major" % weights.name,
        output=weights.output.copy_as_batch_major())
    weights_remaining_axes, weights_squeeze_axes, _ = self._weights_remaining_axes(
      base=self.base.output, weights=weights.output, auto_squeeze=auto_squeeze,
      exception_prefix=repr(self))
    if weights_squeeze_axes:
      weights = InternalLayer(
        network=weights.network, name="%s_squeezed" % weights.name,
        output=weights.output.copy_squeeze_axes(weights_squeeze_axes))
      weights_remaining_axes, weights_squeeze_axes, _ = self._weights_remaining_axes(
        base=self.base.output, weights=weights.output, auto_squeeze=auto_squeeze,
        exception_prefix="%r after squeeze" % self)
      assert not weights_squeeze_axes
    weights_axis_to_reduce = self._weights_time_axis_to_reduce(weights=weights.output, base=self.base.output)

    weights_data = weights.output.copy_as_batch_major()
    weights_data = weights_data.copy_move_axis(
      self._weights_time_axis_to_reduce(weights=weights_data, base=self.base.output), 1)  # (B,T,...)
    self.base_weights = weights_data.placeholder
    del weights_data

    # Do not duplicate the same/similar code as DotLayer, i.e. just use it here.
    # We have weights on the left-side and base on the right side of the matmul,
    # because we want to end up with the base feature as the right-most outer axis,
    # and the axis to be reduced is the right-most time dim of weights,
    # which likely is already the overall right-most axis because of SoftmaxOverSpatialLayer,
    # i.e. exactly as we need it.
    self.dot_layer = DotLayer(
      name="%s_dot" % self.name,
      network=self.network,
      output=self.output,
      sources=[weights, self.base],
      red1=weights.output.get_description_from_axis(weights_axis_to_reduce), red2="T",
      var1=[weights.output.get_description_from_axis(a) for a in weights_remaining_axes], var2="F")
    self.output = self.dot_layer.output

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    return super(GenericAttentionLayer, self).get_dep_layers() + [self.weights]

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    d.setdefault("from", [])
    super(GenericAttentionLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["weights"] = get_layer(d["weights"])

  @classmethod
  def _weights_time_axis_to_reduce(cls, weights, base):
    """
    :param Data weights:
    :param Data base:
    :return: axis
    :rtype: int
    """
    # Note: This is tricky. The old behavior was to just use time_dim_axis.
    # In some cases, it might make sense to use the last dynamic axis.
    # If we had SoftmaxOverSpatialLayer before, we should make sure to use that same axis.
    # SoftmaxOverSpatialLayer by default uses time_dim_axis.
    # We also should maybe check that this matches the base time dim axis.
    dyn_axes = weights.get_dynamic_axes()
    # for static time-dim
    if weights.time_dim_axis not in dyn_axes:
      return weights.time_dim_axis
    assert dyn_axes, "no dynamic axes in %r" % weights
    # Simple case: Only one dynamic axis.
    # Do not do any further checks in this case. The runtime will crash if non-matching and this is simple to identify.
    if len(dyn_axes) == 1:
      assert dyn_axes == [weights.time_dim_axis]
      return weights.time_dim_axis
    # Other case: Template construction, so we might not have access to the dim tag info.
    # (Yet, at least. This might change when we improve the dim tag handling.)
    if not weights.size_placeholder:
      # At template construction, it should not matter anyway.
      assert weights.time_dim_axis in dyn_axes
      return weights.time_dim_axis
    # New behavior: Require that we have a matching time dim, and use that one.
    base_time_tag = base.get_dim_tag(base.time_dim_axis)
    matched_dyn_axes = [axis for axis in dyn_axes if weights.get_dim_tag(axis).is_equal(base_time_tag)]
    if len(matched_dyn_axes) > 1:
      # Ok, this case is again tricky
      # (but also kind of artificial, as you usually do not have this case;
      # it happens only in some of the test cases).
      # If there was SoftmaxOverSpatialLayer before, it would have used the time_dim_axis,
      # so if we have that one, use it.
      if weights.time_dim_axis in matched_dyn_axes:
        return weights.time_dim_axis
      # Just take the last. Not sure what else to do.
      return matched_dyn_axes[-1]
    if len(matched_dyn_axes) == 1:
      return matched_dyn_axes[0]
    from pprint import pformat
    raise Exception(
      ("no matching time axis found in weights %r with dim tags\n%s;\n"
       "base %r with time dim tag\n %r") % (
        weights, pformat(weights.get_batch_shape_dim_tags()), base, base_time_tag))

  @classmethod
  def _weights_remaining_axes(cls, base, weights, auto_squeeze, exception_prefix):
    """
    :param Data base:
    :param Data weights:
    :param bool auto_squeeze: auto-squeeze any weight-axes with dim=1 away
    :param str exception_prefix:
    :return:
      list of remaining axes from weights (which we keep in the output),
      list of weight squeeze axes,
      list of common pairs (weights axis, base axis)
    :rtype: (list[int], list[int], list[(int,int)])
    """
    base_rem_axes = base.get_axes(exclude_batch=True, exclude_time=True)
    base_rem_axes.remove(base.feature_dim_axis)
    weights_rem_axes = weights.get_axes(exclude_batch=True)
    weights_axis_to_reduce = cls._weights_time_axis_to_reduce(weights=weights, base=base)
    assert weights.batch_shape[weights_axis_to_reduce] == base.batch_shape[base.time_dim_axis]
    weights_rem_axes.remove(weights_axis_to_reduce)
    weights_squeeze_axes = []
    common_axes = [(weights.batch_dim_axis, base.batch_dim_axis)]
    for weights_rem_axis in list(reversed(weights_rem_axes)):
      if base_rem_axes:
        if weights.batch_shape[weights_rem_axis] == base.batch_shape[base_rem_axes[-1]]:
          common_axes.append((weights_rem_axis, base_rem_axes[-1]))
          base_rem_axes.pop(-1)
          weights_rem_axes.remove(weights_rem_axis)
          continue
      if auto_squeeze:
        if weights.batch_shape[weights_rem_axis] == 1:
          weights_rem_axes.remove(weights_rem_axis)
          weights_squeeze_axes.append(weights_rem_axis)
          continue
    assert not base_rem_axes, (
      ("%s: We assume that from the base (%r), we reduce the time axis, keep the feature axis,"
       " and have all others matching with the weights (%r)."
       " However, we have these remaining base axes which do not match: %r."
       " We have these remaining weights axes: %r.") % (
        exception_prefix, base, weights, base_rem_axes, weights_rem_axes))
    return weights_rem_axes, weights_squeeze_axes, common_axes

  @classmethod
  def get_out_data_from_opts(cls, base, weights, auto_squeeze=True, sources=(), **kwargs):
    """
    :param LayerBase base:
    :param LayerBase weights:
    :param bool auto_squeeze:
    :param list[LayerBase] sources: ignored, should be empty (checked in __init__)
    :rtype: Data
    """
    from .basic import DotLayer, InternalLayer
    if not weights.output.is_batch_major:
      weights = InternalLayer(
        network=weights.network, name="%s_batch_major" % weights.name,
        output=weights.output.copy_template().copy_as_batch_major())
    weights_remaining_axes, weights_squeeze_axes, _ = cls._weights_remaining_axes(
      base=base.output, weights=weights.output, auto_squeeze=auto_squeeze,
      exception_prefix="%s %r" % (cls.__name__, kwargs["name"]))
    if weights_squeeze_axes:
      weights = InternalLayer(
        network=weights.network, name="%s_squeezed" % weights.name,
        output=weights.output.copy_template().copy_squeeze_axes(weights_squeeze_axes))
      weights_remaining_axes, weights_squeeze_axes, _ = cls._weights_remaining_axes(
        base=base.output, weights=weights.output, auto_squeeze=auto_squeeze,
        exception_prefix="%s %r after squeeze" % (cls.__name__, kwargs["name"]))
      assert not weights_squeeze_axes
    weights_axis_to_reduce = cls._weights_time_axis_to_reduce(weights=weights.output, base=base.output)
    return DotLayer.get_out_data_from_opts(
      sources=[weights, base],
      red1=weights.output.get_description_from_axis(weights_axis_to_reduce), red2="T",
      var1=[weights.output.get_description_from_axis(a) for a in weights_remaining_axes], var2="F",
      **kwargs)


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
    from returnn.tf.util.basic import expand_multiple_dims
    with tf.name_scope("att_energy"):
      # Get base of shape (batch, base_time, inner).
      base = self.base.output.get_placeholder_as_batch_major()  # (batch, base_time, n_out)
      base_seq_lens = self.base.output.get_sequence_lengths()
      base_ctx = self.base_ctx.output.get_placeholder_as_batch_major()  # (batch, base_time, inner)
      # Get source of shape (batch, inner, 1).
      source = tf.expand_dims(self.input_data.placeholder, axis=1)  # (batch, 1, inner)
      energy_in = tf.tanh(base_ctx + source)  # (batch, base_time, inner)
      energy_weights = self.add_param(tf_compat.v1.get_variable("v", shape=(self.input_data.dim,)))  # (inner,)
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
    from returnn.tf.util.basic import expand_dims_unbroadcast, dimshuffle

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
      # We don't have multi_batch_beam for TF yet.
      # But tf.gather_nd or so might anyway be better to use here.
      # If that will not result in a sparse gradient in the while-loop,
      # some slicing with min(idxs)..max(idxs) might be anther option to at least reduce it a bit.
      # Note that gather_nd is broken up to TF 1.0 for this use case (see test_TFUtil.py),
      # so you need TF >=1.1 here.
      from returnn.tf.util.basic import assert_min_tf_version
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
    """
    :param int|None inner_size:
    :rtype: Data
    """
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
  The attention will be over the time-dimension.
  If there is no time-dimension, we expect to be inside a :class:`RecLayer`;
  also, this is only valid with `attention_to_past_only=True`.

  See also `dot_product_attention` here:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
  """
  layer_class = "self_attention"
  recurrent = True

  def __init__(self, num_heads, total_key_dim,
               key_shift=None,
               forward_weights_init="glorot_uniform", attention_dropout=0.0,
               attention_left_only=False, initial_state=None, restrict_state_to_last_seq=False,
               state_var_lengths=None, **kwargs):
    """
    :param int num_heads:
    :param int total_key_dim: i.e. key_dim == total_key_dim // num_heads
    :param LayerBase|None key_shift: additive term to the key. can be used for relative positional encoding.
      Should be of shape (num_queries,num_keys,key_dim), currently without batch-dimension.
      I.e. that should be shape (1,t,key_dim) inside rec-layer or (T,T,key_dim) outside.
    :param str forward_weights_init: see :func:`returnn.tf.util.basic.get_initializer`
    :param float attention_dropout:
    :param bool attention_left_only: will mask out the future. see Attention is all you need.
    :param str|float|int|None initial_state: see RnnCellLayer.get_rec_initial_state_inner().
    :param bool restrict_state_to_last_seq: see code comment below
    :param None|tf.Tensor|()->tf.Tensor state_var_lengths:
      if passed, a Tensor containing the number of keys in the state_var for
      each batch-entry, used for decoding in RASR.
    """
    super(SelfAttentionLayer, self).__init__(**kwargs)
    self._restrict_state_to_last_seq = restrict_state_to_last_seq
    input_data = self.input_data.copy_as_batch_major().copy_with_feature_last()
    assert self._rec_previous_layer or input_data.time_dim_axis is not None, (
      "%s: This layer is expected to be used inside a RecLayer, or to have input with time." % self)
    total_value_dim = self.output.dim
    assert total_key_dim % num_heads == 0, "must be divisible"
    assert total_value_dim % num_heads == 0, "must be divisible. total_value_dim = n_out"
    from returnn.tf.util.basic import get_initializer, dot, get_shape, to_int32_64, where_bc
    batch_dim = input_data.get_batch_dim()
    with self.var_creation_scope():
      fwd_weights_initializer = get_initializer(
        forward_weights_init, seed=self.network.random.randint(2 ** 31), eval_local_ns={"layer": self})
      n_in = input_data.dim
      mat_n_out = total_key_dim * 2 + total_value_dim  # Q, K, V
      mat = self.add_param(tf_compat.v1.get_variable(
        name="QKV", shape=(n_in, mat_n_out), dtype=tf.float32, initializer=fwd_weights_initializer),
        axes_split_info=[[n_in], [total_key_dim, total_key_dim, total_value_dim]])
      prev_mask = None
      if self._rec_previous_layer:
        assert input_data.time_dim_axis is None
        assert attention_left_only
        # (batch,heads,time,k-dim//heads)
        prev_k_left = self._rec_previous_layer.rec_vars_outputs["k_left"]
        # (batch,heads,time,v-dim//heads)
        prev_v_left = self._rec_previous_layer.rec_vars_outputs["v_left"]
        prev_mask = self._rec_previous_layer.rec_vars_outputs.get("mask_left")
      else:
        assert input_data.time_dim_axis is not None
        if initial_state is not None:
          prev_k_left = RnnCellLayer.get_rec_initial_state_inner(
            initial_state=initial_state, name=self.name, rec_layer=self,
            state_key="k_left",
            initial_shape=(batch_dim, num_heads, 0, total_key_dim // num_heads),
            shape_invariant=(None, num_heads, None, total_key_dim // num_heads))
          prev_v_left = RnnCellLayer.get_rec_initial_state_inner(
            initial_state=initial_state, name=self.name, rec_layer=self,
            state_key="v_left",
            initial_shape=(batch_dim, num_heads, 0, total_value_dim // num_heads),
            shape_invariant=(None, num_heads, None, total_value_dim // num_heads))
        else:
          prev_k_left, prev_v_left = None, None
    x = input_data.placeholder
    if input_data.sparse:
      x = tf.nn.embedding_lookup(mat, to_int32_64(x))
    else:
      x = dot(x, mat)
    x.set_shape(tf.TensorShape(input_data.batch_shape_dense[:-1] + (mat_n_out,)))
    x_shape = [-1, -1, num_heads, mat_n_out // num_heads]  # without time
    if input_data.time_dim_axis is None:
      assert input_data.batch_dim_axis == 0
      x_shape[1] = 1
    else:
      assert input_data.time_dim_axis in (0, 1)
    assert input_data.batch_dim_axis in (0, 1)
    x_shape[input_data.batch_dim_axis] = batch_dim
    x = tf.reshape(x, x_shape)  # (batch,time|1)|(time|1,batch) + (heads,qkv-dim//heads)
    x.set_shape(tf.TensorShape([None, None, num_heads, mat_n_out // num_heads]))
    assert input_data.batch_dim_axis in (0, 1)
    # (batch,heads,time|1,qkv-dim//heads)
    x = tf.transpose(x, [input_data.batch_dim_axis, 2, 1 - input_data.batch_dim_axis, 3])
    x.set_shape((None, num_heads, None, mat_n_out // num_heads))
    q, k, v = tf.split(
      x, [total_key_dim // num_heads, total_key_dim // num_heads, total_value_dim // num_heads], axis=-1, name="qkv")
    # (batch,heads,time|1,{q,k,v}-dim//heads)
    q.set_shape((None, num_heads, None, total_key_dim // num_heads))
    k.set_shape((None, num_heads, None, total_key_dim // num_heads))
    v.set_shape((None, num_heads, None, total_value_dim // num_heads))
    q *= (total_key_dim // num_heads) ** -0.5
    orig_k = k
    orig_q = q
    have_prev_kv_left = (prev_k_left is not None)
    assert have_prev_kv_left == (prev_v_left is not None)
    if have_prev_kv_left:
      # Memory for kv.
      self.rec_vars_outputs["k_left"] = k  # usually will be overwritten by the new k below
      self.rec_vars_outputs["v_left"] = v  # usually will be overwritten by the new v below
      k = tf.concat([prev_k_left, k], axis=2)
      v = tf.concat([prev_v_left, v], axis=2)
      k.set_shape((None, num_heads, None, total_key_dim // num_heads))
      v.set_shape((None, num_heads, None, total_value_dim // num_heads))
      if restrict_state_to_last_seq:
        # 'Last' means the current `k`/`v` here, before the concat with `prev_k_left` / `prev_v_left`.
        # I.e. we wont update `rec_vars_outputs` to the concatenated variant; it will exclude `prev_k_left` and
        # `prev_v_left`. Note that this means a difference depending whether we are inside the loop or not.
        # If we are inside the loop, we should update until the end of the seq, and then restrict to the last seq.
        # This is handled in post_process_final_rec_vars_outputs.
        # Otherwise just leave `rec_vars_outputs` as it is already.
        if self._rec_previous_layer:
          self.rec_vars_outputs["k_left"] = k
          self.rec_vars_outputs["v_left"] = v
      else:  # this is usually the case
        self.rec_vars_outputs["k_left"] = k
        self.rec_vars_outputs["v_left"] = v
    # Dot-attention. Resulting last time dimension will be used to perform the softmax over, and will the be reduced.
    # (batch,heads,num_queries|1,num_keys) e.g. (batch,heads,time|1,time)
    energy = tf.matmul(q, k, transpose_b=True, name="energy")
    if key_shift:
      # We could add it to `k`, but instead, to avoid unbroadcasting, we do it as an additional matmul.
      # key_shift expected to be of shape (num_queries|1,num_keys,key_dim).
      key_shift_data = key_shift.output
      assert key_shift_data.batch_dim_axis is None and key_shift_data.dim == total_key_dim // num_heads
      k_ = key_shift_data.placeholder
      # See also _relative_attention_inner here: https://github.com/tensorflow/tensor2tensor
      q_t = tf.transpose(q, [2, 0, 1, 3])  # [num_queries|1,batch,heads,key_dim]
      q_t_r = tf.reshape(
        q_t, [tf.shape(q_t)[0], batch_dim * num_heads, total_key_dim // num_heads])  # [num_queries|1,batch*heads,k-dim]
      with tf.control_dependencies([tf_compat.v1.assert_equal(
            message="check_shape_of_key_shift:",
            x=tf.shape(k_), y=[tf.shape(q_t)[0], tf.shape(energy)[-1], total_key_dim // num_heads])]):
        energy_ = tf.matmul(q_t_r, k_, transpose_b=True)  # [num_queries|1,batch*heads,num_keys]
      energy_ = tf.reshape(
        energy_, [tf.shape(q_t)[0], batch_dim, num_heads, tf.shape(energy)[-1]])  # [num_queries|1,batch,heads,num_keys]
      energy_ = tf.transpose(energy_, [1, 2, 0, 3])  # [batch,heads,num_queries|1,num_keys]
      energy += energy_
    if input_data.time_dim_axis is not None:
      if attention_left_only:
        # We also ignore the input data sequence length, because we expect that frames outside the seq length
        # are anyway ignored.
        from returnn.tf.util.basic import matrix_triangular
        num_queries = tf.shape(orig_q)[2]
        num_keys = tf.shape(orig_k)[2]
        # (1,1,num_queries,num_keys)
        energy_mask = matrix_triangular((1, 1, num_queries, num_keys), dtype=tf.bool, lower=True)
        if have_prev_kv_left:
          energy_mask_left = tf.ones((1, 1, num_queries, tf.shape(prev_k_left)[2]), dtype=tf.bool)
          energy_mask = tf.concat([energy_mask_left, energy_mask], axis=-1)
      else:
        energy_mask = tf.sequence_mask(
          input_data.get_sequence_lengths(), maxlen=tf.shape(energy)[-1])  # (batch,time)
        energy_mask = tf.reshape(energy_mask, [tf.shape(energy)[0], 1, 1, tf.shape(energy)[-1]])  # (batch,1,1,time)
      if state_var_lengths is not None and have_prev_kv_left:
        if callable(state_var_lengths):
          state_var_lengths = state_var_lengths()
        assert isinstance(state_var_lengths, tf.Tensor)
        state_var_max_length = tf.shape(prev_k_left)[-2]
        total_max_length = tf.shape(energy)[-1]
        inverted_prefix_mask = tf.sequence_mask(
          state_var_max_length - state_var_lengths, maxlen=total_max_length, name="inverted_prefix_mask")
        # shape: (batch,1,1,time)
        inverted_prefix_mask = tf.reshape(inverted_prefix_mask, [tf.shape(energy)[0], 1, 1, tf.shape(energy)[-1]])
        energy_mask = tf.math.logical_xor(energy_mask, inverted_prefix_mask)
      energy = where_bc(energy_mask, energy, float("-inf") * tf.ones_like(energy), name="energy_masked")
    elif prev_mask is not None:
      assert self._rec_previous_layer and have_prev_kv_left
      mask = tf.concat([prev_mask, tf.ones([batch_dim, 1], dtype=tf.bool)], axis=1)  # B,T
      mask.set_shape((None, None))
      self.rec_vars_outputs["mask_left"] = mask
      energy = where_bc(
        mask[:, None, None, :],  # (B,1(H),1(num_queries),num_keys|T)
        energy, float("-inf") * tf.ones_like(energy), name="energy_masked_history")
      self._assign_masked_comp_state_transform_func_callback(k, v, mask)
    weights = tf.nn.softmax(energy, name="weights")  # (batch,heads,num_queries|time,num_keys|time)
    if attention_dropout:
      weights = self.network.cond_on_train(
        fn_train=lambda: tf_util.dropout(
          weights,
          keep_prob=1 - attention_dropout,
          seed=self.network.random.randint(2 ** 31)),
        fn_eval=lambda: weights)
    v = tf.matmul(weights, v, name="reduce_att")  # (batch,heads,time,v-dim//heads)
    v.set_shape(tf.TensorShape([None, num_heads, None, total_value_dim // num_heads]))
    v = tf.transpose(v, [0, 2, 1, 3])  # (batch,time,heads,v-dim//heads)
    v = tf.reshape(v, get_shape(v)[:2] + [total_value_dim], name="merge_vdim")  # (batch,time,v-dim)
    v.set_shape(tf.TensorShape([None, None, total_value_dim]))
    if input_data.time_dim_axis is None:
      # Squeeze away the time-dim, which should be 1.
      v = tf.squeeze(v, axis=1)
    self.output.placeholder = v
    self.output.size_placeholder = input_data.size_placeholder.copy()

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    super(SelfAttentionLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if d.get("key_shift", None):
      d["key_shift"] = get_layer(d["key_shift"])

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
    out = sources[0].output.copy_template(name="%s_output" % name)
    out = out.copy_as_batch_major().copy_with_feature_last()
    batch_dim_tag = out.dim_tags[out.batch_dim_axis]
    feat_tag = FeatureDim("%s_self_att_feat" % name, dimension=n_out, auto_generated=True)
    if len(out.shape_dense) >= 2:
      if all(out.shape_dense[:-1]):
        time_dim = int(numpy.prod(out.shape[:-1]))
      else:
        time_dim = None
      time_tag = Dim(
        kind=Dim.Types.Spatial, description="%s_self_att_time" % name, dimension=time_dim, auto_generated=True)
      dim_tags = (batch_dim_tag, time_tag, feat_tag)
    else:
      dim_tags = (batch_dim_tag, feat_tag)
    data_opts = out.get_kwargs(include_special_axes=False)
    data_opts["dim_tags"] = dim_tags
    data_opts["dtype"] = "float32"
    data_opts.pop("sparse", None)
    data_opts.pop("vocab", None)
    data_opts.pop("dim", None)
    return Data(**data_opts)

  # noinspection PyMethodOverriding
  @classmethod
  def get_rec_initial_extra_outputs(cls, batch_dim, rec_layer, network, num_heads, total_key_dim, n_out, name,
                                    initial_state=None, sources=(), **kwargs):
    """
    :param tf.Tensor batch_dim:
    :param RecLayer|LayerBase rec_layer:
    :param returnn.tf.network.TFNetwork network:
    :param int num_heads:
    :param int total_key_dim:
    :param int n_out:
    :param str name:
    :param str|float|int|None initial_state:
    :param list[LayerBase] sources:
    :rtype: dict[str, tf.Tensor]
    """
    data = get_concat_sources_data_template(sources)
    data = data.copy_as_batch_major()
    if data.time_dim_axis is None or initial_state is not None:
      total_value_dim = n_out
      # Assume inside RecLayer, or initial_state set explicitly.
      # Before, we used a tf.TensorArray.
      # However, that has higher memory consumptions than just using a tensor and concatenating to it.
      # Still, this is not ideal as we create a new tensor containing the previous t-1 keys/values for every time step
      # t, thus requiring quadratic memory usage.
      # (batch,heads,time,k-dim//heads)
      k_left = RnnCellLayer.get_rec_initial_state_inner(
        initial_state=initial_state, name=name, rec_layer=rec_layer,
        state_key="k_left",
        initial_shape=(batch_dim, num_heads, 0, total_key_dim // num_heads),
        shape_invariant=(None, num_heads, None, total_key_dim // num_heads))
      # (batch,heads,time,v-dim//heads)
      v_left = RnnCellLayer.get_rec_initial_state_inner(
        initial_state=initial_state, name=name, rec_layer=rec_layer,
        state_key="v_left",
        initial_shape=(batch_dim, num_heads, 0, total_value_dim // num_heads),
        shape_invariant=(None, num_heads, None, total_value_dim // num_heads))
      res = {"k_left": k_left, "v_left": v_left}
      if network.is_extra_internal_template_construction():
        # This could be inside MaskedComputationLayer or some other conditional execution layer.
        # Need mask for history.
        res["mask_left"] = tf.ones((batch_dim, 0), dtype=tf.bool)  # B,T
      return res
    return {}

  # noinspection PyMethodOverriding
  @classmethod
  def get_rec_initial_extra_outputs_shape_invariants(cls, rec_layer, sources, network,
                                                     num_heads, total_key_dim, n_out, **kwargs):
    """
    :param returnn.tf.layers.rec.RecLayer|LayerBase|None rec_layer: for the scope
    :param list[LayerBase] sources:
    :param returnn.tf.network.TFNetwork network:
    :param int num_heads:
    :param int total_key_dim:
    :param int n_out:
    :rtype: dict[str, tf.TensorShape]
    """
    data = get_concat_sources_data_template(sources)
    data = data.copy_as_batch_major()
    if data.time_dim_axis is None:
      # Assume inside RecLayer. See get_rec_initial_extra_outputs.
      total_value_dim = n_out
      res = {
        "k_left": tf.TensorShape((None, num_heads, None, total_key_dim // num_heads)),
        "v_left": tf.TensorShape((None, num_heads, None, total_value_dim // num_heads))}
      if network.is_extra_internal_template_construction():
        res["mask_left"] = tf.TensorShape((None, None))
      return res
    return {}

  def post_process_final_rec_vars_outputs(self, rec_vars_outputs, seq_len):
    """
    :param dict[str,tf.Tensor] rec_vars_outputs:
    :param tf.Tensor seq_len: shape (batch,)
    :rtype: dict[str,tf.Tensor]
    """
    if self.input_data.time_dim_axis is None and self._restrict_state_to_last_seq:
      # k_left and v_left should be of shape (batch, heads, time, {k,v}_dim_per_head).
      # time will be >= max(seq_len); could be more if we use e.g. initial_state=keep_over_epoch.
      rec_vars_outputs["k_left"] = rec_vars_outputs["k_left"][:, :, -tf.reduce_max(seq_len):]
      rec_vars_outputs["v_left"] = rec_vars_outputs["v_left"][:, :, -tf.reduce_max(seq_len):]
    return rec_vars_outputs

  @classmethod
  def _assign_masked_comp_state_transform_func_callback(cls, k, v, mask):
    """
    :param tf.Tensor k: [B,H,T,K]
    :param tf.Tensor v: [B,H,T,V]
    :param tf.Tensor mask: shape [B,T]
    """
    # noinspection PyUnusedLocal,PyShadowingNames
    def _masked_transform_mask(value, prev_value, mask):
      """
      :param tf.Tensor value: mask [B,T]
      :param tf.Tensor prev_value: mask [B,T']
      :param tf.Tensor mask: [B]
      :return: masked value
      :rtype: tf.Tensor
      """
      return tf.concat([prev_value, mask[:, None]], axis=1)

    # noinspection PyUnusedLocal,PyShadowingNames
    def _masked_transform_kv(value, prev_value, mask):
      """
      :param tf.Tensor value: mask [B,H,T,K|V]
      :param tf.Tensor prev_value: mask [B,H,T',K|V]
      :param tf.Tensor mask: [B]
      :return: masked value
      :rtype: tf.Tensor
      """
      return value

    mask.RETURNN_masked_comp_state_transform = _masked_transform_mask
    k.RETURNN_masked_comp_state_transform = _masked_transform_kv
    v.RETURNN_masked_comp_state_transform = _masked_transform_kv


class PositionalEncodingLayer(_ConcatInputLayer):
  """
  Provides positional encoding in the form of (batch, time, n_out) or (time, batch, n_out)
  where n_out is the number of channels, if it is run outside a :class:`RecLayer`,
  and (batch, n_out) or (n_out, batch)
  if run inside a :class:`RecLayer`, where it will depend on the current time frame.

  Assumes one source input with a time dimension if outside a :class:`RecLayer`.
  With `add_to_input`, it will calculate `x + input`, and the output shape is the same as the input

  The positional encoding is the same as in Tensor2Tensor.
  See :func:`returnn.tf.util.basic.get_positional_encoding`.
  """
  layer_class = "positional_encoding"
  recurrent = True

  def __init__(self, axis=NotSpecified, add_to_input=False, constant=-1, offset=None, **kwargs):
    """
    :param Dim|str|NotSpecified axis: if not specified, check for time_dim_axis, otherwise assume rec step
    :param bool add_to_input: will add the signal to the input
    :param int constant: if positive, always output the corresponding positional encoding.
    :param None|LayerBase offset: Specify the offset to be added to positions. Expect shape (batch, time) or (batch,).
    """
    from returnn.tf.util.data import single_step_dim
    super(PositionalEncodingLayer, self).__init__(**kwargs)
    assert len(self.sources) == 1, "%s: expect a single source" % self
    source = self.input_data
    assert source.feature_dim_axis == source.batch_ndim - 1  # Must be last
    if add_to_input:
      assert source.dim == self.output.dim
    assert self.output.feature_dim_axis == self.output.batch_ndim - 1  # Must be last
    output_templ_wo_feat = self.output.copy_template_excluding_axis(self.output.feature_dim_axis)
    offset_data = None
    if offset:
      offset_data = offset.output.copy_compatible_to(output_templ_wo_feat, check_dtype=False)
    from returnn.tf.util.basic import get_positional_encoding
    if axis is NotSpecified:
      if source.have_time_axis():
        axis = "T"
      else:
        axis = single_step_dim
    if axis != single_step_dim:
      src_axis_int = source.get_axis_from_description(axis, allow_int=False)
      out_axis_int = self.output.get_axis_from_description(axis, allow_int=False)
      if constant > -1:
        position = constant * tf.ones([1] * output_templ_wo_feat.batch_ndim, tf.int32)
        if offset_data:
          position += offset_data.placeholder  # (batch, len)
        # signal has shape (1, len) or (batch, len) or (1, 1) or more ones
        signal = get_positional_encoding(num_channels=self.output.dim, position=position)
        if not add_to_input and not offset_data:  # Need to tile the time dimension
          tiles = [1] * self.output.batch_ndim
          tiles[out_axis_int] = tf_util.get_shape_dim(source.placeholder, src_axis_int)
          signal = tf.tile(signal, tiles)
      else:
        length = tf_util.get_shape_dim(source.placeholder, src_axis_int)
        position = tf.range(length)  # (len,)
        # Expand dims e.g. (1, len)
        position = tf.reshape(
          position,
          [length if i == out_axis_int else 1 for i in range(output_templ_wo_feat.batch_ndim)])
        if offset_data:
          position += offset_data.placeholder
        # signal has shape (1,len,n_out) or (batch,len,n_out)
        signal = get_positional_encoding(num_channels=self.output.dim, position=position)
    else:  # single step
      if constant > -1:
        position = tf.convert_to_tensor([constant])
      else:
        position = tf.convert_to_tensor([self.network.get_rec_step_index()])
      if offset_data:
        position += offset_data.placeholder  # (batch,)
      signal = get_positional_encoding(num_channels=self.output.dim, position=position)  # (1,n_out) or (batch,n_out)

    if add_to_input:
      signal += source.placeholder
    else:
      # tile to batch dimension explicitly, as batch_dim=1 will not be automatically unbroadcasted
      tiles = [1] * self.output.batch_ndim
      tiles[self.output.batch_dim_axis] = self.get_batch_dim()
      signal = tf.tile(signal, tiles)
    self.output.placeholder = signal

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param ((str)->LayerBase) get_layer:
    """
    if d.get("from", None) is None:
      if network.is_inside_rec_layer():
        d["from"] = [":i"]
      else:
        d["from"] = ["data"]
    super(PositionalEncodingLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if d.get("offset", None):
      d["offset"] = get_layer(d["offset"])

  @classmethod
  def get_out_data_from_opts(cls, name, network, add_to_input=False, sources=(), **kwargs):
    """
    :param str name:
    :param returnn.tf.network.TFNetwork network:
    :param bool add_to_input:
    :param list[LayerBase] sources:
    :rtype: Data
    """
    assert len(sources) > 0, "%s %r: must have one source" % (cls, name)
    if add_to_input:
      return get_concat_sources_data_template(sources, name="%s_output" % name)  # just the same as the input
    return super(PositionalEncodingLayer, cls).get_out_data_from_opts(
      name=name, network=network, sources=sources, **kwargs)


class KenLmStateLayer(_ConcatInputLayer):
  """
  Get next word (or subword) each frame,
  accumulates string,
  keeps state of seen string so far,
  returns score (+log space, natural base e) of sequence,
  using KenLM (https://kheafield.com/code/kenlm/) (see :mod:`TFKenLM`).
  EOS (</s>) token must be used explicitly.
  """
  layer_class = "kenlm"
  recurrent = True

  def __init__(self, lm_file, vocab_file=None, vocab_unknown_label="UNK", bpe_merge_symbol=None,
               axis=NotSpecified,
               input_step_offset=0, dense_output=False,
               debug=False,
               **kwargs):
    """
    :param str|()->str lm_file: ARPA file or so. whatever KenLM supports
    :param str|None vocab_file: if the inputs are symbols, this must be provided. see :class:`Vocabulary`
    :param str vocab_unknown_label: for the vocabulary
    :param str|None bpe_merge_symbol: e.g. "@@" if you want to apply BPE merging
    :param Dim|str|NotSpecified axis:
    :param int input_step_offset: if provided, will consider the input only from this step onwards
    :param bool dense_output: whether we output the score for all possible succeeding tokens
    :param bool debug: prints debug info
    """
    if callable(lm_file):
      lm_file = lm_file()
    import returnn.tf.util.ken_lm as tf_ken_lm
    from returnn.tf.util.data import single_step_dim
    from returnn.tf.util.basic import expand_multiple_dims
    super(KenLmStateLayer, self).__init__(**kwargs)
    if axis is NotSpecified:
      if self.input_data.have_time_axis():
        axis = "T"
      else:
        axis = single_step_dim
    # Note: We later could extend it and have the state-behavior just as the :class:`CumsumLayer`.
    assert axis == single_step_dim and self._rec_previous_layer and self.input_data.time_dim_axis is None, (
      "%s: currently expected to run inside rec layer" % self)
    # Create KenLM handle. Use var scope to explicitly have it outside the loop.
    with self.var_creation_scope():
      self.lm_handle = tf_ken_lm.ken_lm_load(filename=lm_file)
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
        from returnn.datasets.util.vocabulary import Vocabulary
        from returnn.tf.network import set_custom_post_init
        self.vocab = Vocabulary(vocab_file=vocab_file, unknown_label=vocab_unknown_label)
        assert self.input_data.sparse and self.vocab.num_labels == self.input_data.dim
        self.tf_vocab = tf_compat.v1.get_variable(
          name="vocab", shape=(self.vocab.num_labels,), dtype=tf.string, trainable=False,
          initializer=tf_compat.v1.zeros_initializer())
        self.add_param(self.tf_vocab, saveable=False, trainable=False)
        set_custom_post_init(var=self.tf_vocab, func=self.vocab.tf_get_init_variable_func(var=self.tf_vocab))
    if input_dtype.is_integer:  # assume word-id in vocab
      assert self.tf_vocab is not None, "%s: provide vocab_file" % self
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
      assert self.tf_vocab is not None, "%s: provide vocab_file" % self
      new_abs_scores, new_abs_scores_dense = tf_ken_lm.ken_lm_abs_score_bpe_strings_dense(
        handle=self.lm_handle,
        bpe_merge_symbol=bpe_merge_symbol or "",
        strings=next_strings,
        labels=self.tf_vocab)
      new_abs_scores_bc = expand_multiple_dims(
        new_abs_scores, [i + new_abs_scores.get_shape().ndims for i in range(self.tf_vocab.get_shape().ndims)])
      new_rel_scores = new_abs_scores_dense - new_abs_scores_bc
    else:
      new_abs_scores = tf_ken_lm.ken_lm_abs_score_bpe_strings(
        handle=self.lm_handle,
        bpe_merge_symbol=bpe_merge_symbol or "",
        strings=next_strings)
      new_rel_scores = new_abs_scores - prev_scores
    if debug:
      # Print some info. Only for the first 3 steps because it will spam a lot.
      from returnn.tf.util.basic import py_print
      new_rel_scores = tf.cond(tf.less_equal(prev_step, 2), lambda: py_print(new_rel_scores, [
        str(self), "; step: ", prev_step,
        "; input shape: ", tf.shape(self.input_data.placeholder), str(self.input_data),
        "; input: ", self.input_data.placeholder,
        "; strings shape: ", tf.shape(next_strings),
        "; strings: ", "'" + next_strings + "'", "; new_abs_scores: ", new_abs_scores,
        "; sparse rel scores: ", new_abs_scores - prev_scores,
        "; min/max/mean rel scores: ",
        tf.reduce_min(new_rel_scores), "/", tf.reduce_max(new_rel_scores), "/", tf.reduce_mean(new_rel_scores)] +
        ["; vocab: ", self.tf_vocab] if self.tf_vocab is not None else []),
        lambda: new_rel_scores)
    self.rec_vars_outputs["scores"] = new_abs_scores
    self.output.placeholder = new_rel_scores

  @classmethod
  def get_out_data_from_opts(cls, name, sources,
                             vocab_file=None, vocab_unknown_label="UNK", dense_output=False,
                             **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str|None vocab_file:
    :param str vocab_unknown_label:
    :param bool dense_output:
    :rtype: Data
    """
    from ..util.data import Dim
    data = get_concat_sources_data_template(sources)
    dtype = tf.as_dtype(data.dtype)
    assert isinstance(dtype, tf.DType)
    assert (data.sparse and dtype.is_integer) or dtype == tf.string
    data = data.copy(name="%s_output" % name)
    data.dtype = "float32"
    data.sparse_dim = None
    if dense_output:
      from returnn.datasets.util.vocabulary import Vocabulary
      vocab = Vocabulary(vocab_file=vocab_file, unknown_label=vocab_unknown_label)
      tag = Dim(
        kind=Dim.Types.Feature, description="%s_ken_lm_vocab" % name,
        dimension=vocab.num_labels, vocab=vocab, auto_generated=True)
      data = data.copy_add_dim_by_tag(tag, axis=-1, unbroadcast=True)
    return data

  @classmethod
  def get_rec_initial_extra_outputs(cls, batch_dim, rec_layer, sources=(), **kwargs):
    """
    :param tf.Tensor batch_dim:
    :param RecLayer|LayerBase rec_layer:
    :param list[LayerBase] sources:
    :rtype: dict[str,tf.Tensor]
    """
    data = get_concat_sources_data_template(sources)
    # Assume inside RecLayer.
    assert all(data.shape)
    batch_shape = data.get_batch_shape(batch_dim=batch_dim)
    return {
      "state": tf.zeros(batch_shape, dtype=tf.string),
      "step": tf.constant(0, dtype=tf.int32),
      "scores": tf.zeros(batch_shape, dtype=tf.float32)}


class EditDistanceLayer(LayerBase):
  """
  Edit distance, also known as Levenshtein distance,
  or in case of words, word error rate (WER),
  or in case of characters, character error rate (CER).

  This will not normalize the result, i.e. return the absolut minimal number of edits
  (add, delete, replace) to transform the first string into the second string.
  For WER/CER, it is common to normalize by the length of the target string,
  but accumulated per epoch.
  """
  layer_class = "edit_distance"
  recurrent = True

  def __init__(self, a, b, a_spatial_dim=None, b_spatial_dim=None, **kwargs):
    """
    :param LayerBase a:
    :param LayerBase b:
    :param str|Dim|None a_spatial_dim:
    :param str|Dim|None b_spatial_dim:
    """
    super(EditDistanceLayer, self).__init__(**kwargs)
    self.a = a
    self.b = b
    if a_spatial_dim is None:
      assert a.output.have_time_axis()
      a_axis = a.output.time_dim_axis
    else:
      a_axis = a.output.get_axis_from_description(a_spatial_dim)
    if b_spatial_dim is None:
      assert b.output.have_time_axis()
      b_axis = b.output.time_dim_axis
    else:
      b_axis = b.output.get_axis_from_description(b_spatial_dim)
    a_template = self.output.copy_template().copy_add_dim_by_tag(
      a.output.dim_tags[a_axis], unbroadcast=True, axis=-1)
    b_template = self.output.copy_template().copy_add_dim_by_tag(
      b.output.dim_tags[b_axis], unbroadcast=True, axis=-1)
    a_ext = a.output.copy_compatible_to(a_template, check_sparse=False)
    b_ext = b.output.copy_compatible_to(b_template, check_sparse=False)
    a_tensor = a_ext.placeholder
    b_tensor = b_ext.placeholder
    from returnn.tf.native_op import edit_distance
    common_shape = None
    if a_ext.batch_ndim != 2:
      common_shape = tf_util.get_shape(a_ext.placeholder)
      common_shape, a_rem_dim = common_shape[:-1], common_shape[-1]
      b_rem_dim = tf_util.get_shape_dim(b_ext.placeholder, -1)
      common_reduce = tf_util.optional_mul(*common_shape)
      if common_reduce is None:
        common_reduce = 1
      a_tensor = tf.reshape(a_tensor, [common_reduce, a_rem_dim])
      b_tensor = tf.reshape(b_tensor, [common_reduce, b_rem_dim])
    y_tensor = edit_distance(
      a=a_tensor,
      a_len=a_ext.dim_tags[-1].dyn_size,
      b=b_tensor,
      b_len=b_ext.dim_tags[-1].dyn_size)
    if a_ext.batch_ndim != 2:
      y_tensor = tf.reshape(y_tensor, common_shape)
    self.output.placeholder = y_tensor

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    return [self.a, self.b]

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param returnn.tf.network.GetLayer|((str)->LayerBase) get_layer: function to get or construct another layer
    """
    d.setdefault("from", [])
    super(EditDistanceLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["a"] = get_layer(d["a"])
    d["b"] = get_layer(d["b"])

  @classmethod
  def get_out_data_from_opts(cls, name, a, b, a_spatial_dim=None, b_spatial_dim=None, **kwargs):
    """
    :param str name:
    :param LayerBase a:
    :param LayerBase b:
    :param str|Dim|None a_spatial_dim:
    :param str|Dim|None b_spatial_dim:
    :rtype: Data
    """
    if a_spatial_dim is None:
      assert a.output.have_time_axis()
      a_axis = a.output.time_dim_axis
    else:
      a_axis = a.output.get_axis_from_description(a_spatial_dim)
    if b_spatial_dim is None:
      assert b.output.have_time_axis()
      b_axis = b.output.time_dim_axis
    else:
      b_axis = b.output.get_axis_from_description(b_spatial_dim)
    a_rem = a.output.copy_template_excluding_axis(a_axis)
    b_rem = b.output.copy_template_excluding_axis(b_axis)
    out = Data.get_common_data([a_rem, b_rem], allow_broadcast_all_sources=False)
    out = out.copy_template("%s_output" % name)
    out.sparse_dim = None
    return out


class EditDistanceTableLayer(LayerBase):
  """
  Given a source and a target, calculates the edit distance table between them.
  Source can be inside a recurrent loop.
  It uses :func:`TFNativeOp.next_edit_distance_row`.

  Usually, if you are inside a rec layer, and "output" is the :class:`ChoiceLayer`,
  you would use "from": "output"
  and "target": "layer:base:data:target" (make sure it has the time dimension).

  See also :class:`OptimalCompletionsLayer`.
  """
  layer_class = "edit_distance_table"
  recurrent = True

  def __init__(self, axis=NotSpecified, debug=False, blank_idx=None, out_dim=None, **kwargs):
    """
    :param Dim|str|NotSpecified axis:
    :param bool debug:
    :param int|None blank_idx: if given, will keep the same row for this source label
    :param Dim|None out_dim:
    """
    from returnn.tf.util.data import single_step_dim
    from returnn.tf.util.basic import is_axis_from_description_recurrent
    super(EditDistanceTableLayer, self).__init__(out_dim=out_dim, **kwargs)
    assert len(self.sources) == 1, "%s: expects exactly a single source" % self
    source_data = self.sources[0].output
    assert source_data.dtype == "int32" and source_data.batch_ndim <= 2 and source_data.sparse
    assert self.target, "%s: 'target' must be set" % self
    target_data = self._get_target_value()
    assert target_data, "%s: target %r not found?" % (self, self.target)
    assert target_data.dtype == "int32" and target_data.batch_ndim == 2 and target_data.have_time_axis()
    dim = target_data.dim
    if blank_idx is not None:
      dim = max(dim, blank_idx + 1)
    assert target_data.sparse and source_data.dim == dim
    target_data = target_data.copy_as_batch_major()
    self._target_data = target_data
    if axis is NotSpecified:
      if source_data.have_time_axis():
        axis = "T"
      else:
        axis = single_step_dim
    if not is_axis_from_description_recurrent(axis=axis, network=self.network, data=source_data):
      raise NotImplementedError(
        "%s: operating on a whole sequence (dim %s in input %s) not supported yet. " % (self, axis, source_data) +
        "Make sure this layer stays inside the recurrent loop.")
    assert source_data.batch_ndim == 1
    # Assume we are inside a rec loop.
    assert self.network.have_rec_step_info()
    self._last_row = self._rec_previous_layer.rec_vars_outputs["state"]
    rec_step_info = self.network.get_rec_step_info()
    batch_dim = self.get_batch_dim()
    mask_flag = rec_step_info.get_prev_end_flag(target_search_choices=self.get_search_choices())
    source = source_data.placeholder
    if blank_idx is None:
      source_len = tf_util.expand_dims_unbroadcast(rec_step_info.step, axis=0, dim=batch_dim)
    else:
      source_len = self._rec_previous_layer.rec_vars_outputs["source_len"]
      mask_flag = tf.logical_or(mask_flag, tf.equal(source, blank_idx))
    from returnn.tf.native_op import next_edit_distance_row
    self._next_row = next_edit_distance_row(
      last_row=self._last_row,
      a=source, a_n=source_len,
      a_ended=mask_flag,
      b=target_data.placeholder, b_len=target_data.get_sequence_lengths())
    if blank_idx is not None:
      self.rec_vars_outputs["source_len"] = source_len + tf_util.where_bc(mask_flag, 0, 1)
    if debug:
      print_out = [str(self)]
      choice = self.get_search_choices()
      if choice:
        print_out += [
          "choice", choice.owner.name,
          "src_beams", choice.src_beams if choice.src_beams is not None else "None"]
      print_out += [
        "a_n", rec_step_info.step,
        "a_ended", rec_step_info.get_prev_end_flag(target_search_choices=self.get_search_choices()),
        "a", tf_util.vocab_idx_repr(source_data.placeholder, target_data),
        "b", tf_util.vocab_idx_repr(target_data.placeholder, target_data),
        "b_len", target_data.get_sequence_lengths(),
        "last_row", self._last_row, "next_row", self._next_row]
      self._next_row = tf_util.py_print(self._next_row, print_out)
    self.rec_vars_outputs["state"] = self._next_row
    self._reduce_out = None  # see get_sub_layer
    self.output.placeholder = self._next_row

  # noinspection PyMethodOverriding
  @classmethod
  def get_rec_initial_extra_outputs(cls, batch_dim, rec_layer, sources, name, target, network, **kwargs):
    """
    :param tf.Tensor batch_dim: for this layer, might be with beam
    :param returnn.tf.layers.rec.RecLayer rec_layer:
    :param list[LayerBase] sources:
    :param str name:
    :param str target:
    :param returnn.tf.network.TFNetwork network:
    :rtype: dict[str,tf.Tensor]
    """
    assert len(sources) == 1, "%s %r: expects exactly a single source" % (cls.__name__, name)
    source_data = sources[0].output
    if source_data.time_dim_axis is not None:
      return {}
    # expects inside rec layer
    assert target, "%s %r: 'target' must be set" % (cls.__name__, name)
    target_data = cls._static_get_target_value(target=target, network=network)
    assert target_data, "target %r not found?" % target
    n_time = tf.shape(target_data.placeholder)[target_data.time_dim_axis]
    d = {"state": tf_util.expand_dims_unbroadcast(tf.range(n_time + 1), axis=0, dim=batch_dim)}
    if kwargs.get("blank_idx", None) is not None:
      d["source_len"] = tf.zeros((batch_dim,), dtype=tf.int32)
    return d

  @classmethod
  def get_rec_initial_output(cls, **kwargs):
    """
    :rtype: tf.Tensor
    """
    initial_extra = cls.get_rec_initial_extra_outputs(**kwargs)
    return initial_extra["state"]

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    d.setdefault("n_out", None)  # avoid the default NotSpecified behavior, because we use target differently
    super(EditDistanceTableLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

  @classmethod
  def get_out_data_from_opts(cls, name, sources, target, network, _target_layers=None, blank_idx=None,
                             out_dim=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param returnn.tf.network.TFNetwork network:
    :param str target:
    :param dict[str,LayerBase] _target_layers:
    :param int|None blank_idx:
    :param Dim|None out_dim:
    :rtype: Data
    """
    assert len(sources) == 1, "%s %r: expects exactly a single source" % (cls.__name__, name)
    source_data = sources[0].output
    assert target, "%s %r: 'target' must be set" % (cls.__name__, name)
    target_data = cls._static_get_target_value(target=target, _target_layers=_target_layers, network=network)
    assert target_data, "target %r not found?" % target
    in_dim = target_data.get_time_dim_tag()
    if not out_dim:
      out_dim = in_dim + 1
    else:
      (in_dim + 1).declare_same_as(out_dim)
    return Data(
      name="%s_output" % name,
      dim_tags=(
        ([source_data.get_batch_dim_tag()] if source_data.have_batch_axis() else []) +
        ([source_data.get_time_dim_tag()] if source_data.have_time_axis() else []) +
        [out_dim]),
      dtype="int32", beam=SearchBeam.get_combined_beam(source_data.beam, target_data.beam))


class OptimalCompletionsLayer(LayerBase):
  """
  We expect to get the inputs from :class:`EditDistanceTableLayer`, esp from the prev frame, like this:
  "opt_completions": {"class": "optimal_completions", "from": "prev:edit_dist_table"}.

  You can also then define this further layer:
  "opt_completion_soft_targets": {
    "class": "eval", "eval": "tf.nn.softmax(tf.cast(source(0), tf.float32))",
    "from": "opt_completions", "out_type": {"dtype": "float32"}},
  and use that as the :class:`CrossEntropyLoss` soft targets
  for the input of the "output" :class:`ChoiceLayer`, e.g. "output_prob".
  This makes most sense when you enable beam search (even, or esp, during training).
  Note that you probably want to have this all before the last choice, where you still have more beams open.
  """
  layer_class = "optimal_completions"
  recurrent = True

  def __init__(self, debug=False, blank_idx=None, **kwargs):
    """
    :param bool debug:
    :param int|None blank_idx:
    """
    super(OptimalCompletionsLayer, self).__init__(**kwargs)
    src_layer, = self.sources
    assert isinstance(src_layer, LayerBase)
    source_data = src_layer.output
    assert source_data.dtype == "int32" and source_data.batch_ndim == 2
    assert source_data.batch_shape == (None, None) and source_data.is_batch_major
    last_row = source_data.placeholder
    assert self.target, "%s: 'target' must be set" % self
    target_data = self._get_target_value()
    assert target_data, "%s: target %r not found?" % (self, self.target)
    assert target_data.dtype == "int32" and target_data.batch_ndim == 2 and target_data.have_time_axis()
    from returnn.tf.native_op import next_edit_distance_reduce
    successors = tf.expand_dims(tf.range(self.output.dim), axis=0)  # [1,dim]
    rec_step_info = self.network.get_rec_step_info()
    src_len = rec_step_info.step
    if blank_idx is not None:
      # We need the correct source len.
      # This is currently a simple way, which expects it to come from EditDistanceTableLayer.
      # Can easily be extended if needed...
      assert "source_len" in src_layer.rec_vars_outputs
      src_len = src_layer.rec_vars_outputs["source_len"]
    reduce_out = next_edit_distance_reduce(
      last_row=last_row,
      a=successors, a_n=src_len,
      a_ended=rec_step_info.get_prev_end_flag(target_search_choices=self.get_search_choices()),
      b=target_data.placeholder, b_len=target_data.get_sequence_lengths(),
      optimal_completion=True, a_blank_idx=blank_idx)
    reduce_out.set_shape((None, self.output.dim))
    if debug:
      from returnn.tf.util.basic import py_print, vocab_idx_repr
      print_out = [str(self)]
      choice = self.get_search_choices()
      if choice:
        print_out += [
          "choice", choice.owner.name,
          "src_beams", choice.src_beams if choice.src_beams is not None else "None"]
      top_values, top_indices = tf.nn.top_k(-reduce_out, k=5)  # (batch,K)
      top_values = -top_values
      print_out += [
        "a_n", rec_step_info.step,
        "a_ended", rec_step_info.get_prev_end_flag(target_search_choices=self.get_search_choices()),
        "a best", vocab_idx_repr(top_indices, target_data), top_values,
        "b", vocab_idx_repr(target_data.placeholder, target_data),
        "b_len", target_data.get_sequence_lengths(),
        "last_row", last_row]
      reduce_out = py_print(reduce_out, print_out)
    self.output.placeholder = reduce_out

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d:
    :param returnn.tf.network.TFNetwork network:
    :param get_layer:
    """
    super(OptimalCompletionsLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    if d.get("n_out", NotSpecified) is not NotSpecified:
      blank_idx = d.get("blank_idx", None)
      if blank_idx is not None:
        d["n_out"] = max(d["n_out"], blank_idx + 1)

  @classmethod
  def get_out_data_from_opts(cls, name, sources, target, network, _target_layers=None, blank_idx=None, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str target:
    :param dict[str,LayerBase] _target_layers:
    :param int|None blank_idx:
    :param returnn.tf.network.TFNetwork network:
    :rtype: Data
    """
    assert len(sources) == 1, "%s %r: expects exactly a single source" % (cls.__name__, name)
    source_data = sources[0].output
    assert target, "%s %r: 'target' must be set" % (cls.__name__, name)
    target_data = cls._static_get_target_value(target=target, _target_layers=_target_layers, network=network)
    assert target_data, "target %r not found?" % target
    assert target_data.dtype == "int32" and target_data.batch_ndim == 2 and target_data.have_time_axis()
    assert target_data.sparse
    dim = target_data.dim
    if blank_idx is not None:
      dim = max(dim, blank_idx + 1)
    return Data(
      name="%s_output" % name,
      shape=(dim,), dim=dim, dtype="int32", sparse=False, time_dim_axis=None,
      beam=SearchBeam.get_combined_beam(source_data.beam, target_data.beam))


class MaskedComputationLayer(LayerBase):
  """
  Given some input [B,T,D] and some mask [B,T] (True or False), we want to perform a computation
  only on the masked frames.
  I.e. let T' be the max seq len of the masked seq, then the masked input would be [B,T',D].
  (This masked input sequence could be calculated via ``tf.boolean_mask`` or ``tf.gather_nd``.)
  The output is [B,T',D'], i.e. we do not undo the masking.
  You are supposed to use :class:`UnmaskLayer` to undo the masking.

  The computation also works within a rec layer, i.e. the input is just [B,D] and the mask is just [B].
  In that case, if the mask is True, it will perform the computation as normal,
  and if it is False, it will just copy the prev output, and also hidden state.
  """
  layer_class = "masked_computation"
  recurrent = True

  def __init__(self, mask, unit, masked_from,
               _layer_class, _layer_desc,
               in_spatial_dim=None,
               out_spatial_dim=None,
               _queried_sub_layers=None,
               _parent_layer_cache=None,
               **kwargs):
    """
    :param LayerBase|None mask:
    :param dict[str] unit:
    :param LayerBase|None masked_from:
    :param Dim|None in_spatial_dim:
    :param Dim|None out_spatial_dim: the masked dim
    :param type[LayerBase] _layer_class:
    :param dict[str] _layer_desc:
    :param dict[str,(Data,type,dict[str])]|None _queried_sub_layers:
    :param dict[str,LayerBase]|None _parent_layer_cache:
    """
    from returnn.tf.network import get_layer_class
    from .base import WrappedInternalLayer
    from returnn.tf.util.basic import where_bc, get_shape, nd_indices
    from tensorflow.python.util import nest
    super(MaskedComputationLayer, self).__init__(**kwargs)
    self.mask = mask
    self.masked_from = masked_from
    self.in_spatial_dim = in_spatial_dim
    self.out_spatial_dim = out_spatial_dim
    if _parent_layer_cache is None:
      _parent_layer_cache = {}
    self.parent_layer_cache = _parent_layer_cache

    sub_layers = {}  # type: typing.Dict[str,LayerBase]
    new_size, new_time, idxs = None, None, None
    if mask:
      if self.network.is_inside_rec_layer():
        assert mask.output.shape == () and mask.output.dtype == "bool", (
          "%s: invalid mask %s (inside rec loop)" % (self, mask))
      else:
        assert mask.output.have_time_axis() and mask.output.shape == (None,) and mask.output.dtype == "bool", (
          "%s: invalid mask %s (outside rec loop)" % (self, mask))
        assert in_spatial_dim and out_spatial_dim
        mask_data = mask.output.copy_as_time_major()
        mask_t = where_bc(mask_data.placeholder, mask_data.get_sequence_mask(), tf.convert_to_tensor(False))
        idxs = tf.cumsum(tf.cast(mask_t, tf.int32), axis=0)  # [T,B] -> idx in T' + 1
        if masked_from:
          new_size = masked_from.output.get_sequence_lengths()
        else:
          new_size = idxs[-1]  # [B]
          out_spatial_dim.get_for_batch_ctx(self.output.batch, self.output.control_flow_ctx).dyn_size = new_size
        new_time = tf.reduce_max(new_size)  # T'
        idxs = where_bc(mask_t, idxs - 1, new_time)

    # noinspection PyShadowingNames
    def get_masked_layer(source):
      """
      :param LayerBase source:
      :rtype: LayerBase
      """
      assert isinstance(source, LayerBase)
      assert mask
      if self.network.is_inside_rec_layer():
        # We can just leave it as-is. The state will handled below.
        return source
      else:
        assert in_spatial_dim
        if in_spatial_dim not in source.output.dim_tags:
          return source
        axis = source.output.get_axis_from_description(in_spatial_dim)
        source_data = source.output.copy_move_axis(old_axis=axis, new_axis=0)  # time-major
        source_data.time_dim_axis = 0
        assert source_data.is_same_time_dim(mask_data), "%s mask and source time dim do not match" % self
        tmp_shape = get_shape(source_data.placeholder)
        tmp_shape[0] = new_time + 1  # one more for the padded data
        res = tf.scatter_nd(nd_indices(idxs, batch_axis=1), source_data.placeholder, shape=tmp_shape)
        res_data = source_data.copy_template().copy_template_replace_dim_tag(axis=0, new_dim_tag=out_spatial_dim)
        res_data.placeholder = res[:new_time]
        res_data.beam = SearchBeam.get_combined_beam(res_data.beam, mask.output.beam)
        layer_desc = dict(base_layer=source, network=source.network, name=source.name, output=res_data)
        layer = WrappedInternalLayer(**layer_desc)
        layer.post_init(layer_desc)
        layer.sources.extend([source, mask])  # add deps
        return layer

    if masked_from:
      assert not self.network.is_inside_rec_layer()
      source_data = masked_from.output.copy(
        name="%s_%s_masked_input" % (masked_from.output.name, self.output.name))
      source_data.available_for_inference = True  # we would make sure that this works at inference
      layer_desc = dict(base_layer=masked_from, network=masked_from.network, name=masked_from.name, output=source_data)
      source = WrappedInternalLayer(**layer_desc)
      source.post_init(layer_desc)
      source.sources.append(masked_from)  # add dep
      sub_layers["data"] = source

    elif len(self.sources) >= 1:
      assert len(self.sources) == 1
      sub_layers["data"] = get_masked_layer(self.sources[0])

    def sub_get_layer(sub_layer_name):
      """
      :param str sub_layer_name:
      :rtype: LayerBase
      """
      if sub_layer_name in sub_layers:
        return sub_layers[sub_layer_name]
      if _parent_layer_cache and sub_layer_name in _parent_layer_cache:
        layer = _parent_layer_cache[sub_layer_name]
      else:
        layer = self.network.get_layer(sub_layer_name)
      # noinspection PyShadowingNames
      source = get_masked_layer(layer)
      sub_layers[sub_layer_name] = source
      return source

    inside_rec_time_dim = self.network.get_inside_rec_time_dim(inside_loop=True)
    over_rec_time_dim = self.network.get_inside_rec_time_dim(inside_loop=False)
    extra_net = self.network.make_extra_net(
      prefix_name="extra._internal.masked(%s)" % self.name, boundary=True)
    if in_spatial_dim == over_rec_time_dim and over_rec_time_dim and not inside_rec_time_dim:
      # Optimized out, so any sub layers will effectively operate on the out spatial dim.
      extra_net._over_rec_time_dim = out_spatial_dim
    layer_desc = unit.copy()
    class_name = layer_desc.pop("class")
    layer_class = get_layer_class(class_name)
    layer_desc["_network"] = extra_net
    layer_desc["_name"] = self.name
    layer_desc["encapsulate"] = True
    layer_desc["rec_previous_layer"] = self._rec_previous_layer
    if self.name_scope is not None and "name_scope" in layer_desc:
      raise Exception("%s: set name_scope only in the masked_computation layer itself" % self)
    layer_desc.setdefault("name_scope", self.name_scope)
    layer_class.transform_config_dict(layer_desc, network=extra_net, get_layer=sub_get_layer)
    # noinspection PyProtectedMember
    layer_desc = extra_net._create_layer_layer_desc(name=self.name, layer_desc=layer_desc)
    sub_out_template = self.output.copy_template(name="%s_output" % self.name)
    if in_spatial_dim and in_spatial_dim in sub_out_template.dim_tags:  # could happen when optimized
      assert out_spatial_dim
      axis = sub_out_template.get_axis_from_description(in_spatial_dim)
      sub_out_template = sub_out_template.copy_template_replace_dim_tag(axis=axis, new_dim_tag=out_spatial_dim)
    sub_out_template = layer_class.fixup_out_data(output=sub_out_template, **layer_desc)
    layer_desc["output"] = sub_out_template

    self.sub_layer = layer_class(**layer_desc)
    assert isinstance(self.sub_layer, LayerBase)
    self.sub_layer.post_init(layer_desc)
    self.sub_layer.output.sanity_check()
    assert self.sub_layer.output.placeholder is not None
    self.rec_vars_outputs = self.sub_layer.rec_vars_outputs.copy()
    self.params = self.sub_layer.params

    self._out_layers = {}  # type: typing.Dict[str,LayerBase]
    out_layer = self.get_sub_layer("")
    self.output = out_layer.output.copy(name="%s_output" % self.name)

    if self.mask and self.network.is_inside_rec_layer():
      assert self._rec_previous_layer
      assert self.mask.output.shape == () and self.mask.output.batch_shape == (None,)
      mask_t = self.mask.output.placeholder

      for key, value in sorted(self.rec_vars_outputs.items()):
        assert isinstance(key, str)
        prev_value = self._rec_previous_layer.rec_vars_outputs[key]
        nest.assert_same_structure(value, prev_value)
        value_flat = nest.flatten(value)
        prev_value_flat = nest.flatten(prev_value)
        assert len(value_flat) == len(prev_value_flat)
        res = []
        mask_shape = tf.shape(mask_t)
        for value_, prev_value_ in zip(value_flat, prev_value_flat):
          assert isinstance(value_, tf.Tensor) and isinstance(prev_value_, tf.Tensor)
          if hasattr(value_, "RETURNN_masked_comp_state_transform"):
            res.append(value_.RETURNN_masked_comp_state_transform(value=value_, prev_value=prev_value_, mask=mask_t))
            continue
          check = tf.Assert(
            tf.reduce_all(tf.equal(tf.shape(value_), tf.shape(prev_value_))),
            ["MaskedComputation non equal state shape", mask_shape, tf.shape(value_), tf.shape(prev_value_)],
            summarize=10)
          with tf.control_dependencies([check]):
            res.append(where_bc(
              condition=tf.reshape(mask_t, [-1] + [1] * (value_.shape.ndims - 1)),  # add broadcast dims
              x=value_,
              y=prev_value_))
        self.rec_vars_outputs[key] = nest.pack_sequence_as(value, res)

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    deps = super(MaskedComputationLayer, self).get_dep_layers()
    if self._rec_previous_layer:
      deps.append(self._rec_previous_layer)
    if self.mask:
      deps.append(self.mask)
    if self.masked_from:
      deps.append(self.masked_from)
    deps.extend(self.parent_layer_cache.values())
    deps.extend(self.sub_layer.get_dep_layers())
    return deps

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    d.setdefault("from", ())
    masked_from = d.pop("masked_from", None)
    if masked_from:
      masked_from = get_layer(masked_from)
      d["masked_from"] = masked_from
      # We explicitly do not want to have these as deps.
      d["from"] = []
    else:
      d["masked_from"] = None
    super(MaskedComputationLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    # Just call it for dep resolution.
    parent_layer_cache = d.setdefault("_parent_layer_cache", {})
    d["_layer_class"], d["_layer_desc"], d["in_spatial_dim"], d["out_spatial_dim"] = cls._create_template(
      name=d["_name"], network=network, sources=d["sources"],
      masked_from=masked_from,
      unit=d["unit"],
      in_spatial_dim=d.get("in_spatial_dim", None),
      out_spatial_dim=d.get("out_spatial_dim", None),
      get_layer=get_layer, _parent_layer_cache=parent_layer_cache)
    if masked_from and not parent_layer_cache:
      # We explicitly do not want to have these as deps.
      d["mask"] = None
    else:
      d["mask"] = get_layer(d["mask"])

  # noinspection PyUnusedLocal
  @classmethod
  def _create_template(cls, name, network, sources, masked_from, unit,
                       in_spatial_dim=None, out_spatial_dim=None,
                       get_layer=None, _parent_layer_cache=None, **kwargs):
    """
    :param str name:
    :param returnn.tf.network.TFNetwork network:
    :param list[LayerBase] sources:
    :param LayerBase masked_from:
    :param dict[str] unit:
    :param Dim|None in_spatial_dim:
    :param Dim|None out_spatial_dim: the masked dim
    :param (str)->LayerBase get_layer:
    :param dict[str,LayerBase]|None parent_layer_cache:
    :return: layer_class, layer_desc, in_spatial_dim, out_spatial_dim
    """
    from returnn.tf.network import get_layer_class
    from .base import WrappedInternalLayer
    if not get_layer:
      get_layer = network.get_layer

    inside_rec_time_dim = network.get_inside_rec_time_dim(inside_loop=True)
    over_rec_time_dim = network.get_inside_rec_time_dim(inside_loop=False)
    if over_rec_time_dim and not in_spatial_dim:
      in_spatial_dim = over_rec_time_dim

    class _Locals:  # Python 2, and modify this maybe via sub_get_layer
      in_spatial_dim_ = in_spatial_dim
      out_spatial_dim_ = out_spatial_dim

    extra_net = None
    _layer_cache = {}

    def sub_get_layer(sub_layer_name):
      """
      :param str sub_layer_name:
      :rtype: LayerBase
      """
      if sub_layer_name in _layer_cache:
        return _layer_cache[sub_layer_name]

      if sub_layer_name == "data" and masked_from:
        if _Locals.out_spatial_dim_:
          if _Locals.out_spatial_dim_ not in masked_from.output.dim_tags:
            masked_from.output.get_time_dim_tag().declare_same_as(_Locals.out_spatial_dim_)
        else:
          _Locals.out_spatial_dim_ = masked_from.output.get_time_dim_tag()
        if network.is_inside_rec_layer():
          source_data = (
            masked_from.output
            .copy_template_excluding_time_dim(
              name="%s_%s_masked_input_frame" % (masked_from.output.name, name))
            .copy_template_set_ctx(network.get_control_flow_ctx()))
        else:
          source_data = masked_from.output.copy_template(
            name="%s_%s_masked_input" % (masked_from.output.name, name))
        source_data.available_for_inference = True  # we would make sure that this works at inference
        source = WrappedInternalLayer(
          base_layer=masked_from, network=masked_from.network, name=masked_from.name, output=source_data)
        _layer_cache[sub_layer_name] = source
        return source

      elif sub_layer_name == "data" and len(sources) >= 1:
        assert len(sources) == 1
        layer = sources[0]
      elif _parent_layer_cache and sub_layer_name in _parent_layer_cache:
        layer = _parent_layer_cache[sub_layer_name]
      else:
        assert not sub_layer_name.startswith(extra_net.extra_name_prefix + ":")
        layer = get_layer(sub_layer_name)
        if not layer:
          return None
        if _parent_layer_cache is not None:
          _parent_layer_cache[sub_layer_name] = layer
      source_data = layer.output.copy_template()
      if not _Locals.in_spatial_dim_:
        _Locals.in_spatial_dim_ = source_data.get_time_dim_tag()

      if _Locals.in_spatial_dim_ in source_data.dim_tags:
        axis = source_data.get_axis_from_description(_Locals.in_spatial_dim_)
        source_data = source_data.copy_move_axis(old_axis=axis, new_axis=0)  # time-major
        # Create own time dim tag, to make sure we have some own custom.
        if not _Locals.out_spatial_dim_:
          _Locals.out_spatial_dim_ = Dim(
            kind=Dim.Types.Spatial, description="%s:masked:time" % name,
            derived_from_tag=_Locals.in_spatial_dim_, auto_generated=True)
          if _Locals.in_spatial_dim_ == over_rec_time_dim and over_rec_time_dim and not inside_rec_time_dim:
            if extra_net:
              # Optimized out, so any sub layers will effectively operate on the out spatial dim.
              extra_net._over_rec_time_dim = _Locals.out_spatial_dim_
        source_data = source_data.copy_template_replace_dim_tag(axis=0, new_dim_tag=_Locals.out_spatial_dim_)
        layer = WrappedInternalLayer(
          base_layer=layer, network=layer.network, name=layer.name,
          output=source_data)
      _layer_cache[sub_layer_name] = layer
      return layer

    if sources or masked_from:
      # Make sure to resolve this first, to maybe declare in_spatial_dim/out_spatial_dim.
      sub_get_layer("data")

    extra_net = network.make_extra_net(
      prefix_name="extra._internal_template.masked(%s)" % name,
      # There should be no need by the base get_layer to get back to us.
      # Nothing from the base/parent will access directly into this extra/sub network,
      # so we can safely place a boundary here.
      # Also, any subnet here must be only a template construction.
      # We will redo the full construction including transform_config_dict on the sub layer in __init__.
      only_template=True, boundary=True)
    if _Locals.in_spatial_dim_ == over_rec_time_dim and over_rec_time_dim and not inside_rec_time_dim:
      # Optimized out, so any sub layers will effectively operate on the out spatial dim.
      extra_net._over_rec_time_dim = _Locals.out_spatial_dim_
    layer_desc = unit.copy()
    class_name = layer_desc.pop("class")
    layer_class = get_layer_class(class_name)
    layer_desc["_network"] = extra_net
    layer_desc["_name"] = name
    layer_desc["encapsulate"] = True
    layer_class.transform_config_dict(layer_desc, network=extra_net, get_layer=sub_get_layer)
    # noinspection PyProtectedMember
    layer_desc = extra_net._create_layer_layer_desc(name=name, layer_desc=layer_desc)
    return layer_class, layer_desc, _Locals.in_spatial_dim_, _Locals.out_spatial_dim_

  @classmethod
  def get_out_data_from_opts(cls, network, masked_from=None, in_spatial_dim=None, out_spatial_dim=None, **kwargs):
    """
    :param returnn.tf.network.TFNetwork network:
    :param LayerBase|None masked_from:
    :param Dim|None in_spatial_dim:
    :param Dim|None out_spatial_dim:
    :rtype: Data
    """
    layer_class, layer_desc = kwargs["_layer_class"], kwargs["_layer_desc"]
    assert issubclass(layer_class, LayerBase)
    output = layer_class.get_out_data_from_opts(**layer_desc)
    assert isinstance(output, Data)
    output.sanity_check()
    if out_spatial_dim and out_spatial_dim in output.dim_tags:
      axis = output.get_axis_from_description(out_spatial_dim)
      output = output.copy_template_replace_dim_tag(axis=axis, new_dim_tag=out_spatial_dim)
    inside_rec_time_dim = network.get_inside_rec_time_dim(inside_loop=True)
    over_rec_time_dim = network.get_inside_rec_time_dim(inside_loop=False)
    if network.is_inside_rec_layer() and output.have_batch_axis():
      output = output.copy_as_batch_major()
    elif not masked_from and in_spatial_dim == over_rec_time_dim and over_rec_time_dim and not inside_rec_time_dim:
      # We will automatically unmask again (https://github.com/rwth-i6/returnn/pull/976).
      assert out_spatial_dim and out_spatial_dim in output.dim_tags
      axis = output.get_axis_from_description(out_spatial_dim)
      output = output.copy_template_replace_dim_tag(axis=axis, new_dim_tag=in_spatial_dim)
      output = output.copy_move_axis(old_axis=axis, new_axis=0)  # time-major
    return output

  class _EncapsulatedSubLayer(InternalLayer):
    pass

  @classmethod
  def get_sub_layer_out_data_from_opts(cls, layer_name, parent_layer_kwargs):
    """
    :param str layer_name: name of the sub_layer (right part of '/' separated path)
    :param dict[str] parent_layer_kwargs: kwargs for the parent layer (as kwargs in cls.get_out_data_from_opts())
    :return: Data template, class type of sub-layer, layer opts (transformed)
    :rtype: (Data, type, dict[str])|None
    """
    queried_sub_layers = parent_layer_kwargs.setdefault("_queried_sub_layers", {})
    if layer_name not in queried_sub_layers:
      sub_cls, sub_layer_kwargs = parent_layer_kwargs["_layer_class"], parent_layer_kwargs["_layer_desc"]
      assert issubclass(sub_cls, LayerBase)
      res = sub_cls.get_sub_layer_out_data_from_opts(layer_name=layer_name, parent_layer_kwargs=sub_layer_kwargs)
      if res is None:
        return None
      sub_out, sub_cls, sub_opts = res
      assert isinstance(sub_out, Data)
      assert isinstance(sub_opts, dict)
      sub_opts = sub_opts.copy()
      # fallback to default parent net and name
      sub_opts.pop("network", None)
      sub_opts.pop("name", None)
      res = (sub_out, sub_cls, sub_opts)
      queried_sub_layers[layer_name] = res
    sub_out, sub_cls, sub_opts = queried_sub_layers[layer_name]
    # This is all encapsulated. The outside should not check get_rec_initial_extra_outputs or anything on the layer.
    # We need initial_output for our own get_rec_initial_extra_outputs.
    return sub_out, cls._EncapsulatedSubLayer, {"initial_output": sub_opts.get("initial_output")}

  def get_sub_layer(self, layer_name):
    """
    :param str layer_name: name of the sub_layer (right part of '/' separated path)
    :return: the sub_layer addressed in layer_name or None if no sub_layer exists
    :rtype: LayerBase|None
    """
    if layer_name in self._out_layers:
      return self._out_layers[layer_name]
    if layer_name == "":
      layer = self.sub_layer
    else:
      layer = self.sub_layer.get_sub_layer(layer_name)
      if not layer:
        return None

    from returnn.tf.layers.base import WrappedInternalLayer
    full_layer_name = (self.name + "/" + layer_name) if layer_name else self.name

    # In case we are in rec layer and optimize out of loop and mask over this axis, and don't have masked_from:
    inside_rec_time_dim = self.network.get_inside_rec_time_dim(inside_loop=True)
    over_rec_time_dim = self.network.get_inside_rec_time_dim(inside_loop=False)
    if (
          not self.masked_from and
          over_rec_time_dim == self.in_spatial_dim and
          over_rec_time_dim and not inside_rec_time_dim):
      # We will automatically unmask again (https://github.com/rwth-i6/returnn/pull/976).
      assert self.mask
      opts = dict(
        name="%s(internal-unmask)" % layer.name, mask=self.mask, sources=[layer], network=layer.network)
      opts["output"] = UnmaskLayer.get_out_data_from_opts(**opts)
      out_layer = UnmaskLayer(**opts)
      return WrappedInternalLayer(
        base_layer=out_layer, name=full_layer_name, network=self.network, output=out_layer.output)
    elif self.mask and self.network.is_inside_rec_layer():
      assert self._rec_previous_layer
      assert self.mask.output.shape == () and self.mask.output.batch_shape == (None,)
      assert layer.output.is_batch_major
      prev_out = layer.output.copy_template()
      key = ("_output/" + layer_name) if layer_name else "_output"
      prev_out.placeholder = self._rec_previous_layer.rec_vars_outputs[key]
      prev_out.sanity_check()
      mask_t = self.mask.output.placeholder
      out = layer.output.copy_template()
      out.placeholder = tf_util.where_bc(
        condition=tf.reshape(mask_t, [-1] + [1] * (layer.output.batch_ndim - 1)),  # add broadcast dims
        x=layer.output.placeholder,
        y=prev_out.placeholder)
      self.rec_vars_outputs[key] = out.placeholder
      return WrappedInternalLayer(
        base_layer=layer, output=out,
        network=self.network, name=(self.name + "/" + layer_name) if layer_name else self.name)
    else:
      return WrappedInternalLayer(
        base_layer=layer, name=full_layer_name, network=self.network, output=layer.output)

  def get_constraints_value(self):
    """
    :rtype: tf.Tensor|None
    """
    return self.sub_layer.get_constraints_value()

  @classmethod
  def get_losses(cls, name, network, output, loss=None, reduce_func=None, layer=None, **kwargs):
    """
    :param str name: layer name
    :param returnn.tf.network.TFNetwork network:
    :param Loss|None loss: argument just as for __init__
    :param Data output: the output (template) for the layer
    :param LayerBase|None layer:
    :param ((tf.Tensor)->tf.Tensor)|None reduce_func:
    :param kwargs: other layer kwargs
    :rtype: list[returnn.tf.network.LossHolder]
    """
    from returnn.tf.network import LossHolder
    # See SubnetworkLayer.get_losses as another example.
    losses = super(MaskedComputationLayer, cls).get_losses(
      name=name, network=network, output=output, loss=loss, layer=layer, reduce_func=reduce_func, **kwargs)
    if layer:
      assert isinstance(layer, MaskedComputationLayer)
      sub_layer = layer.sub_layer
      sub_layer_kwargs = sub_layer.kwargs
      sub_layer_class = sub_layer.__class__
    else:
      sub_layer = None
      sub_layer_class, sub_layer_kwargs = kwargs["_layer_class"], kwargs["_layer_desc"]
      assert issubclass(sub_layer_class, LayerBase) and isinstance(sub_layer_kwargs, dict)
      sub_layer_kwargs["output"] = output
    for loss in sub_layer_class.get_losses(reduce_func=reduce_func, layer=sub_layer, **sub_layer_kwargs):
      assert isinstance(loss, LossHolder)
      losses.append(loss.copy_new_base(network=network, name="%s/%s" % (name, loss.name)))
    return losses

  @classmethod
  def get_rec_initial_output(cls, initial_output=None, **kwargs):
    """
    :param initial_output:
    :rtype: tf.Tensor
    """
    assert initial_output is None, "%s %r, should be configured via the unit" % (cls, kwargs["name"])
    d = cls.get_rec_initial_extra_outputs(**kwargs)
    return d["_output"]

  @classmethod
  def get_rec_initial_extra_outputs(cls, batch_dim, rec_layer, **kwargs):
    """
    :param tf.Tensor batch_dim: for this layer, might be with beam
    :param returnn.tf.layers.rec.RecLayer rec_layer:
    :rtype: dict[str,tf.Tensor]
    """
    layer_class, layer_desc = kwargs["_layer_class"], kwargs["_layer_desc"]
    assert issubclass(layer_class, LayerBase) and isinstance(layer_desc, dict)
    name = kwargs["name"]
    output = kwargs["output"]
    assert isinstance(name, str)
    assert isinstance(output, Data)
    assert issubclass(layer_class, LayerBase)
    with layer_class.cls_setup_scope(**kwargs):
      output = output.copy_as_batch_major()
      d = layer_class.get_rec_initial_extra_outputs(
        batch_dim=batch_dim, rec_layer=rec_layer, output=output, **layer_desc)
      initial_out = layer_class.get_rec_initial_output(
        batch_dim=batch_dim, rec_layer=rec_layer, output=output, **layer_desc)
      assert "_output" not in d
      d["_output"] = initial_out
      cell = rec_layer.cell
      assert isinstance(cell, _SubnetworkRecCell)
      for layer_name, layer_templ in cell.layer_data_templates.items():
        if not layer_name.startswith(name + "/"):
          continue
        layer_name = layer_name[len(name) + 1:]
        sub_initial_out = layer_templ.layer_class_type.get_rec_initial_output(
          batch_dim=batch_dim, rec_layer=rec_layer, **layer_templ.kwargs)
        assert "_output/" + layer_name not in d
        d["_output/" + layer_name] = sub_initial_out
      return d

  @classmethod
  def get_rec_initial_extra_outputs_shape_invariants(cls, rec_layer, **kwargs):
    """
    :param returnn.tf.layers.rec.RecLayer rec_layer:
    :return: optional shapes for the tensors by get_rec_initial_extra_outputs
    :rtype: dict[str,tf.TensorShape]
    """
    # Very similar to get_rec_initial_extra_outputs.
    layer_class, layer_desc = kwargs["_layer_class"], kwargs["_layer_desc"]
    assert issubclass(layer_class, LayerBase) and isinstance(layer_desc, dict)
    name = kwargs["name"]
    output = kwargs["output"]
    assert isinstance(name, str)
    assert isinstance(output, Data)
    assert issubclass(layer_class, LayerBase)
    with layer_class.cls_setup_scope(**kwargs):
      d = layer_class.get_rec_initial_extra_outputs_shape_invariants(rec_layer=rec_layer, **layer_desc)
      assert "_output" not in d
      d["_output"] = tf.TensorShape(output.copy_as_batch_major().batch_shape)
      cell = rec_layer.cell
      assert isinstance(cell, _SubnetworkRecCell)
      for layer_name, layer_templ in cell.layer_data_templates.items():
        if not layer_name.startswith(name + "/"):
          continue
        layer_name = layer_name[len(name) + 1:]
        assert "_output/" + layer_name not in d
        d["_output/" + layer_name] = tf.TensorShape(layer_templ.output.copy_as_batch_major().batch_shape)
      return d


class UnmaskLayer(LayerBase):
  """
  This is meant to be used together with :class:`MaskedComputationLayer`,
  which operates on input [B,T,D], and given a mask, returns [B,T',D'].
  This layer :class:`UnmaskLayer` is supposed to undo the masking,
  i.e. to recover the original time dimension, i.e. given [B,T',D'], we output [B,T,D'].
  This is done by repeating the output for the non-masked frames,
  via the last masked frame.

  If this layer is inside a recurrent loop, i.e. we get [B,D'] as input,
  this is a no-op, and we just return the input as is.
  In that case, the repetition logic is handled via :class:`MaskedComputationLayer`.
  """
  layer_class = "unmask"
  recurrent = True

  def __init__(self, mask, **kwargs):
    """
    :param LayerBase mask: the same as as used for :class:`MaskedComputationLayer`.
      Outside loop: [B,T] or [T,B], original T. Inside loop, just [B].
    """
    from returnn.tf.util.basic import concat_with_opt_broadcast, nd_indices, same_control_flow_ctx, where_bc
    super(UnmaskLayer, self).__init__(**kwargs)
    self.mask = mask
    src_layer = self.sources[0]
    batch_dim = self.get_batch_dim()
    orig_time_dim = self.mask.output.get_time_dim_tag() if self.mask.output.have_time_axis() else None
    if not src_layer.output.have_time_axis():
      assert self.network.is_inside_rec_layer()
      assert self.output.placeholder is not None  # should be the copy of source already
      # Nothing needs to be done.
      if self.network.is_inside_rec_layer():
        # We have this state, although we don't need it, we still must set it.
        self.rec_vars_outputs["t"] = tf.zeros([batch_dim], dtype=tf.int32) - 1

    else:  # src has time dim
      # We need the initial output for proper unmasking.
      # It's a bit tricky to get the initial output.
      rec_parent_layer = self.network.get_rec_parent_layer(inside_loop=False)
      src_layer_opts = src_layer.kwargs.copy()
      src_layer_out_inner_template = src_layer.output.copy_template_excluding_time_dim()
      src_layer_opts["output"] = src_layer_out_inner_template
      initial = src_layer.get_rec_initial_output(
        batch_dim=batch_dim, rec_layer=rec_parent_layer, **src_layer_opts)  # [B,D']
      if self.network.is_inside_rec_layer():
        # This UnmaskLayer is inside the rec loop but the source is outside (or at least does not have a time dim).
        # The RecLayer will not unroll the source when the dim tag do not match, i.e. when it is masked.
        with same_control_flow_ctx(src_layer.output.placeholder):
          src = src_layer.output.copy_as_bt_or_tb_major()
        mask_out = self.mask.output
        assert mask_out.shape == () and mask_out.batch_shape == (None,) and mask_out.dtype == "bool", (
          "%s: invalid mask %s (inside rec loop)" % (self, self.mask))
        prev_t = self._rec_previous_layer.rec_vars_outputs["t"]  # [B]
        t = prev_t + tf.cast(mask_out.placeholder, tf.int32)  # [B]
        self.rec_vars_outputs["t"] = t
        idxs_nd = nd_indices(tf.maximum(t, 0), indices_batch_major=src.is_batch_major)  # [B,2]
        y = tf.gather_nd(src.placeholder, idxs_nd)  # [B,D']
        y = where_bc(tf.equal(t, -1)[:, None], initial, y)
        self.output.placeholder = y

      elif orig_time_dim in src_layer.output.dim_tags:  # outside rec loop, and src seems already unmasked
        # Do nothing, as src already seems unmasked.
        # Note that we might run into this case due to: https://github.com/rwth-i6/returnn/pull/976
        self.output.placeholder = src_layer.output.get_placeholder_as_time_major()

      else:  # outside rec loop
        assert src_layer.output.get_time_dim_tag() != self.mask.output.get_time_dim_tag(), (
          "%s: unexpected source" % self)
        src = src_layer.output.copy_as_time_major()
        mask_out = self.mask.output
        assert mask_out.shape == (None,) and mask_out.batch_shape == (None, None) and mask_out.dtype == "bool", (
          "%s: invalid mask %s (outside rec loop)" % (self, self.mask))
        mask_out = mask_out.copy_as_time_major()
        mask_t = mask_out.placeholder  # [T,B], e.g. [1,0,1] (ignoring batch-dim for example)
        idxs = tf.cumsum(tf.cast(mask_t, tf.int32), axis=mask_out.time_dim_axis)  # [T,B], e.g. [1,1,2]
        initial_wt = tf.expand_dims(initial, axis=0)  # add time axis
        src_t = concat_with_opt_broadcast(
          [initial_wt, src.placeholder], allow_broadcast=[True, False], axis=0, name="concat_in_time")  # [T'+1,B,D']
        idxs_nd = nd_indices(idxs, batch_axis=src.batch_dim_axis)  # [T,B,2]
        y = tf.gather_nd(src_t, idxs_nd)  # [T,B,D']
        self.output.placeholder = y

  def get_dep_layers(self):
    """
    :rtype: list[LayerBase]
    """
    deps = super(UnmaskLayer, self).get_dep_layers()
    deps.append(self.mask)
    return deps

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    super(UnmaskLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
    d["mask"] = get_layer(d["mask"])

  @classmethod
  def get_out_data_from_opts(cls, name, network, sources, mask, **kwargs):
    """
    :param str name:
    :param returnn.tf.network.TFNetwork network:
    :param list[LayerBase] sources:
    :param LayerBase mask:
    :rtype: Data
    """
    assert len(sources) == 1
    source, = sources
    assert isinstance(source, LayerBase)
    out = source.output.copy(name="%s_output" % name)
    assert isinstance(out, Data)
    out.beam = SearchBeam.get_combined_beam(out.beam, mask.output.beam)
    if network.is_inside_rec_layer():
      if out.have_time_axis():
        # The masked-computation layer could have been moved out. In that case, it will return some output
        # which is not compatible with the rec layer (because reduced, because of the masking),
        # thus when we unroll it to get into the loop, the RecLayer would have kept it as-is,
        # i.e. it should still have that time-dim-axis.
        # Maybe we should do some extra checks if that is like we assume, but for now, just assume that.
        return out.copy_template_excluding_time_dim().copy_template_set_ctx(network.get_control_flow_ctx())
      return out
    assert out.have_time_axis()
    out = out.copy_as_time_major()
    out.size_placeholder[0] = mask.output.get_sequence_lengths()
    return out

  # noinspection PyMethodOverriding
  @classmethod
  def get_rec_initial_extra_outputs(cls, batch_dim, rec_layer, sources, **kwargs):
    """
    :param tf.Tensor batch_dim: for this layer, might be with beam
    :param returnn.tf.layers.rec.RecLayer rec_layer:
    :param list[LayerBase] sources:
    :rtype: dict[str,tf.Tensor]
    """
    # This is only called if we are inside the rec layer.
    # In that case, we have a state: The running index in our source.
    # Note that if the source is also inside the rec layer, we do not need this.
    # However, there is no way at this point to know this.
    return {"t": tf.zeros([batch_dim], dtype=tf.int32) - 1}


# noinspection PyAbstractClass
class BaseRNNCell(rnn_cell.RNNCell):
  """
  Extends :class:`rnn_cell.RNNCell` by having explicit static attributes describing some properties.
  """

  def get_input_transformed(self, x, batch_dim=None):
    """
    Usually the cell itself does the transformation on the input.
    However, it would be faster to do it outside the recurrent loop.
    This function will get called outside the loop.

    :param tf.Tensor x: (time, batch, dim), or (batch, dim)
    :param tf.Tensor|None batch_dim:
    :return: like x, maybe other feature-dim
    :rtype: tf.Tensor|tuple[tf.Tensor]
    """
    return x


class VanillaLSTMCell(BaseRNNCell):
  """
  Just a vanilla LSTM cell, which is compatible to our NativeLSTM (v1 and v2).
  """

  def __init__(self, num_units):
    """
    :param int num_units:
    """
    super(VanillaLSTMCell, self).__init__()
    self.num_units = num_units

  @property
  def output_size(self):
    """
    :rtype: int
    """
    return self.num_units

  @property
  def state_size(self):
    """
    :rtype: rnn_cell.LSTMStateTuple
    """
    return rnn_cell.LSTMStateTuple(self.num_units, self.num_units)

  def get_input_transformed(self, x, batch_dim=None):
    """
    :param tf.Tensor x: (time, batch, dim), or (batch, dim)
    :param tf.Tensor|None batch_dim:
    :return: like x, maybe other feature-dim
    :rtype: tf.Tensor|tuple[tf.Tensor]
    """
    from returnn.tf.util.basic import dot
    input_dim = x.get_shape().dims[-1].value
    assert input_dim is not None, "%r shape unknown" % (x,)
    weights = tf_compat.v1.get_variable("W", shape=(input_dim, self.num_units * 4))
    bias = tf_compat.v1.get_variable("b", shape=(self.num_units * 4,), initializer=tf.constant_initializer(0.0))
    return dot(x, weights) + bias

  def __call__(self, inputs, state, scope=None):
    """
    Run this RNN cell on inputs given a state.

    :param tf.Tensor inputs:
    :param rnn_cell.LSTMStateTuple state:
    :return: (LSTM output h, LSTM state (consisting of cell state c and output h)
    :rtype: (tf.Tensor, rnn_cell.LSTMStateTuple)
    """
    from tensorflow.python.ops.math_ops import sigmoid, tanh
    from returnn.tf.util.basic import dot, reuse_name_scope
    var_scope_name = tf_compat.v1.get_variable_scope().name
    if var_scope_name.endswith("/rnn"):
      var_scope_name = var_scope_name[:-len("/rnn")]
    with reuse_name_scope(var_scope_name, absolute=True):  # compatibility
      prev_c, prev_h = state
      weights = tf_compat.v1.get_variable("W_re", shape=(self.num_units, self.num_units * 4))
      inputs = inputs + dot(prev_h, weights)
      # NativeLstm2: cell-in + input, forget and output gates
      c_in, i, f, o = tf.split(inputs, num_or_size_splits=4, axis=-1)
      new_c = sigmoid(f) * prev_c + sigmoid(i) * tanh(c_in)
      new_h = sigmoid(o) * tanh(new_c)
      return new_h, rnn_cell.LSTMStateTuple(new_c, new_h)


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
    from returnn.tf.network import TFNetwork
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
    """
    :rtype: int
    """
    return self._num_units

  @property
  def state_size(self):
    """
    :rtype: int
    """
    return self._num_units

  @staticmethod
  def _linear(x, output_dim):
    """
    :param tf.Tensor x:
    :param int output_dim:
    :rtype: tf.Tensor
    """
    from returnn.tf.util.basic import dot
    input_dim = x.get_shape().dims[-1].value
    assert input_dim is not None, "%r shape unknown" % (x,)
    weights = tf_compat.v1.get_variable("W", shape=(input_dim, output_dim))
    x = dot(x, weights)
    return x

  def _get_dropout_mask(self):
    """
    :rtype: tf.Tensor
    """
    if self._dropout_mask is not None:
      return self._dropout_mask

    from returnn.tf.util.basic import default_control_flow_ctx, cond
    # Create the dropout masks outside the loop:
    with default_control_flow_ctx():
      def get_mask():
        """
        :rtype: tf.Tensor
        """
        if self.batch_size is not None:
          batch_size = self.batch_size
        else:
          batch_size = LayerBase.get_recent_layer().get_batch_dim()
        keep_prob = 1.0 - self.dropout
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += tf_compat.v1.random_uniform(
          (batch_size, self._num_units), seed=self.dropout_seed, dtype=tf.float32)
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

  def get_input_transformed(self, x, batch_dim=None):
    """
    :param tf.Tensor x: (time, batch, dim)
    :param tf.Tensor|None batch_dim:
    :return: (time, batch, num_units * 2)
    :rtype: tf.Tensor
    """
    x = self._linear(x, self._num_units * 2)
    bias = tf_compat.v1.get_variable(
      "b", shape=(self._num_units * 2,),
      initializer=tf.constant_initializer(
        [0.0] * self._num_units + [self.transform_bias] * self._num_units))
    x += bias
    return x

  # noinspection PyMethodOverriding
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
      with tf_compat.v1.variable_scope('depth_%i' % i):
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


class _WrapBaseCell(BaseRNNCell):
  """
  Simpler helper wrapper class, for :class:`BaseRNNCell`.
  """
  cell_type = None

  def __init__(self, *args, **kwargs):
    """
    :param int num_units:
    """
    super(_WrapBaseCell, self).__init__()
    self.cell = self.cell_type(*args, **kwargs)
    assert isinstance(self.cell, rnn_cell.RNNCell)
    assert hasattr(self.cell, "get_input_transformed")

  @property
  def output_size(self):
    """
    :rtype: int
    """
    return self.cell.output_size

  @property
  def state_size(self):
    """
    :rtype: int|tuple[int]
    """
    return self.cell.state_size

  def get_input_transformed(self, x, batch_dim=None):
    """
    :param tf.Tensor x: (time, batch, dim), or (batch, dim)
    :param tf.Tensor|None batch_dim:
    :return: like x, maybe other feature-dim
    :rtype: tf.Tensor|tuple[tf.Tensor]
    """
    if x.get_shape().ndims == 2 and batch_dim is None:
      # In that case, we are probably inside the recurrent loop,
      # so the best way to get the batch dim but not depend on `x`:
      batch_dim = LayerBase.get_recent_layer().get_batch_dim()
    return self.cell.get_input_transformed(x, batch_dim=batch_dim)

  # noinspection PyMethodOverriding
  def call(self, inputs, state):
    """
    :param tf.Tensor inputs:
    :param tf.Tensor|tuple[tf.Tensor] state:
    :rtype: tf.Tensor|tuple[tf.Tensor]
    """
    return self.cell.call(inputs, state)


class BlocksparseLSTMCell(_WrapBaseCell):
  """
  Standard LSTM but uses OpenAI blocksparse kernels to support bigger matrices.

  Refs:

    https://blog.openai.com/block-sparse-gpu-kernels/
    https://github.com/openai/blocksparse
    https://s3-us-west-2.amazonaws.com/openai-assets/blocksparse/blocksparsepaper.pdf

  It uses our own wrapper, see :func:`TFNativeOp.init_blocksparse`.
  """

  def __init__(self, *args, **kwargs):
    from returnn.tf.native_op import init_blocksparse
    init_blocksparse(with_native_module=not kwargs.get("always_dense", False))
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from blocksparse.lstm import BlocksparseLSTMCell as CellImpl
    self.cell_type = CellImpl
    kwargs = kwargs.copy()
    if kwargs.get('is_training', None) is None:
      from returnn.tf.network import TFNetwork
      kwargs['is_training'] = TFNetwork.get_current_network().train_flag
    from returnn.tf.util.basic import is_gpu_available_in_session
    if not is_gpu_available_in_session():
      kwargs.setdefault("fast_layer_norm", False)
    super(BlocksparseLSTMCell, self).__init__(*args, **kwargs)

  def call(self, *args, **kwargs):
    """
    :param args: passed to super
    :param kwargs: passed to super
    :rtype: tf.Tensor|tuple[tf.Tensor]
    """
    y = super(BlocksparseLSTMCell, self).call(*args, **kwargs)
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from blocksparse.lstm import BlocksparseLSTMCell as CellImpl
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from blocksparse.matmul import BlocksparseMatMul
    assert isinstance(self.cell, CellImpl)
    print("BlocksparseLSTMCell, matmuls:", file=log.v4)
    for d in self.cell.linear.matmuls:
      bsmm = d["bsmm"]
      if bsmm:
        assert isinstance(bsmm, BlocksparseMatMul)
        print('  %s: sparsity %.4f%%' % (d["weights"], 100.0 - 100.0 * bsmm.sparsity), file=log.v4)
      else:
        print('  %s: dense' % d["weights"], file=log.v4)
    return y

  def load_params_from_native_lstm(self, values_dict, session):
    """
    :param tf.compat.v1.Session session:
    :param dict[str,numpy.ndarray] values_dict:
    """
    assert set(values_dict.keys()) == {"W", "W_re", "b"}
    assert len(self.cell.linear.matmuls) == 2
    m1, m2 = self.cell.linear.matmuls
    assert m1["bsmm"] and m2["bsmm"], 'both sparse'
    w_ff = values_dict["W"]
    w_re = values_dict["W_re"]
    w_b = values_dict["b"]
    assert w_ff.shape[-1] == w_re.shape[-1] == w_b.shape[-1]
    assert w_ff.shape[-1] % 4 == 0
    old_dim = w_ff.shape[-1] // 4
    assert m1["bias"].get_shape().dims[-1].value % 4 == 0
    new_dim = m1["bias"].get_shape().dims[-1].value // 4
    assert new_dim > old_dim
    bsize = m1["bsmm"].bsize
    assert bsize == m2["bsmm"].bsize
    assert new_dim % bsize == 0
    assert m1["bsmm"].KB == new_dim * 4 // bsize
    assert m2["bsmm"].CB == new_dim // bsize
    assert m2["bsmm"].KB == new_dim * 4 // bsize

    for w_old, m in ((w_ff, m1), (w_re, m2)):
      w_new = session.run(m["weights"])
      assert w_new.shape == (m["bsmm"].blocks, bsize, bsize)
      m["bsmm"].np_update_parts(w_new, w_old, last_dim_num_splits=4)
      m["weights"].load(w_new, session=session)

    b_old = w_b
    b_new = session.run(m1["bias"])
    assert b_new.shape == (new_dim * 4,)
    for gate_idx in range(4):
      b_new[gate_idx * new_dim:gate_idx * new_dim + old_dim] = b_old[gate_idx * old_dim:(gate_idx + 1) * old_dim]
    m1["bias"].load(b_new, session=session)


class BlocksparseMultiplicativeMultistepLSTMCell(_WrapBaseCell):
  """
  Multiplicative LSTM with multiple steps, as in the OpenAI blocksparse paper.
  Uses OpenAI blocksparse kernels to support bigger matrices.

  Refs:

    https://blog.openai.com/block-sparse-gpu-kernels/
    https://github.com/openai/blocksparse
    https://s3-us-west-2.amazonaws.com/openai-assets/blocksparse/blocksparsepaper.pdf

  """

  def __init__(self, *args, **kwargs):
    from returnn.tf.native_op import init_blocksparse
    init_blocksparse(with_native_module=not kwargs.get("always_dense", False))
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from blocksparse.lstm import BlocksparseMultiplicativeMultistepLSTMCell as CellImpl
    self.cell_type = CellImpl
    kwargs = kwargs.copy()
    if kwargs.get('is_training', None) is None:
      from returnn.tf.network import TFNetwork
      kwargs['is_training'] = TFNetwork.get_current_network().train_flag
    from returnn.tf.util.basic import is_gpu_available_in_session
    if not is_gpu_available_in_session():
      kwargs.setdefault("fast_layer_norm", False)
    super(BlocksparseMultiplicativeMultistepLSTMCell, self).__init__(*args, **kwargs)

  def call(self, *args, **kwargs):
    """
    :rtype: tf.Tensor
    """
    y = super(BlocksparseMultiplicativeMultistepLSTMCell, self).call(*args, **kwargs)
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from blocksparse.lstm import BlocksparseMultiplicativeMultistepLSTMCell as CellImpl
    assert isinstance(self.cell, CellImpl)
    print("BlocksparseMultiplicativeMultistepLSTMCell, matmuls:", file=log.v4)
    for d in self.cell.linear.matmuls:
      bsmm = d["bsmm"]
      if bsmm:
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        from blocksparse.matmul import BlocksparseMatMul
        assert isinstance(bsmm, BlocksparseMatMul)
        print('  %s: sparsity %.4f%%' % (d["weights"], 100.0 - 100.0 * bsmm.sparsity), file=log.v4)
      else:
        print('  %s: dense' % d["weights"], file=log.v4)
    return y


class LayerNormVariantsLSTMCell(BaseRNNCell):
  """LSTM unit with layer normalization and recurrent dropout

  This LSTM cell can apply different variants of layer normalization:

  1. Layer normalization as in the original paper:
  Ref: https://arxiv.org/abs/1607.06450
  This can be applied by having:
    all default params (global_norm=True, cell_norm=True, cell_norm_in_output=True)

  2. Layer normalization for RNMT+:
  Ref: https://arxiv.org/abs/1804.09849
  This can be applied by having:
    all default params except
    - global_norm = False
    - per_gate_norm = True
    - cell_norm_in_output = False

  3. TF official `LayerNormBasicLSTMCell`
  Ref: https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LayerNormBasicLSTMCell
  This can be reproduced by having:
    all default params except
    - global_norm = False
    - per_gate_norm = True

  4. Sockeye LSTM layer normalization implementations
  Ref: https://github.com/awslabs/sockeye/blob/master/sockeye/rnn.py

  `LayerNormLSTMCell` can be reproduced by having:
    all default params except
    - with_concat = False (just efficiency, no difference in the model)

  `LayerNormPerGateLSTMCell` can be reproduced by having:
    all default params except:
    (- with_concat = False)
    - global_norm = False
    - per_gate_norm = True

  Recurrent dropout is based on:
        https://arxiv.org/abs/1603.05118

  Prohibited LN combinations:
  - global_norm and global_norm_joined both enabled
  - per_gate_norm with global_norm or global_norm_joined

  """

  def __init__(self,
               num_units,
               norm_gain=1.0,
               norm_shift=0.0,
               forget_bias=0.0,
               activation=tf.tanh,
               is_training=None,
               dropout=0.0,
               dropout_h=0.0,
               dropout_seed=None,
               with_concat=False,
               global_norm=True,
               global_norm_joined=False,
               per_gate_norm=False,
               cell_norm=True,
               cell_norm_in_output=True,
               hidden_norm=False,
               variance_epsilon=1e-12):
    """
    :param int num_units: number of lstm units
    :param float norm_gain: layer normalization gain value
    :param float norm_shift: layer normalization shift (bias) value
    :param float forget_bias: the bias added to forget gates
    :param activation: Activation function to be applied in the lstm cell
    :param bool is_training: if True then we are in the training phase
    :param float dropout: dropout rate, applied on cell-in (j)
    :param float dropout_h: dropout rate, applied on hidden state (h) when it enters the LSTM (variational dropout)
    :param int dropout_seed: used to create random seeds
    :param bool with_concat: if True then the input and prev hidden state
      is concatenated for the computation. this is just about computation performance.
    :param bool global_norm: if True then layer normalization is applied
      for the forward and recurrent outputs (separately).
    :param bool global_norm_joined: if True, then layer norm is applied on LSTM in
      (forward and recurrent output together)
    :param bool per_gate_norm: if True then layer normalization is applied
      per lstm gate
    :param bool cell_norm: if True then layer normalization is applied
      to the LSTM new cell output
    :param bool cell_norm_in_output: if True, the normalized cell is also used in the output
    :param bool hidden_norm: if True then layer normalization is applied
      to the LSTM new hidden state output
    """

    super(LayerNormVariantsLSTMCell, self).__init__()
    from returnn.tf.network import TFNetwork
    self._num_units = num_units
    self.norm_grain = norm_gain
    self.norm_shift = norm_shift
    self.forget_bias = forget_bias
    self.activation = activation

    if is_training is None:
      is_training = TFNetwork.get_current_network().train_flag
    self.is_training = is_training

    self.dropout = dropout
    self.dropout_h = dropout_h
    if dropout_seed is None:
      dropout_seed = TFNetwork.get_current_network().random.randint(2 ** 31)
    self.dropout_seed = dropout_seed

    self.with_concat = with_concat

    # used for different layer norm variants
    self.global_norm = global_norm
    self.global_norm_joined = global_norm_joined
    self.per_gate_norm = per_gate_norm
    self.cell_norm = cell_norm
    self.cell_norm_in_output = cell_norm_in_output
    self.hidden_norm = hidden_norm
    self.variance_epsilon = variance_epsilon

    assert not (self.global_norm_joined and self.global_norm), (
      '%s: global_norm and global_norm_joined can not be enabled together' % self)

    assert not (self.per_gate_norm and self.global_norm), (
      '%s: per_gate_norm can not be enabled with global_norm' % self)

    assert not (self.per_gate_norm and self.global_norm_joined), (
      '%s: per_gate_norm can not be enabled with global_norm_joined' % self)

  @property
  def output_size(self):
    """
    :rtype: int
    """
    return self._num_units

  @property
  def state_size(self):
    """
    :rtype: rnn_cell.LSTMStateTuple
    """
    return rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

  def _norm(self, inputs, with_beta=True, add_forget_bias=False, name=None):
    """
    :param tf.Tensor inputs: (B,D), or (T,B,D)
    :param bool with_beta: if True, then add norm shift to the normalized inputs
    :param bool add_forget_bias: if True, then add forget bias to the initializer
    :param str name: variable scope name
    :return: (B,D) or (T,B,D)
    :rtype: tf.Tensor
    """
    assert name is not None
    shape = inputs.get_shape()[-1:]
    gamma_init = tf.constant_initializer(self.norm_grain)
    beta_init = self.norm_shift
    if add_forget_bias and self.forget_bias > 0:
      beta_init += self.forget_bias
    mean, variance = tf_compat.v1.nn.moments(inputs, axes=[-1], keep_dims=True)
    normalized_input = (inputs - mean) * tf_compat.v1.rsqrt(variance + self.variance_epsilon)
    g = tf_compat.v1.get_variable("gamma_" + name, shape=shape, initializer=gamma_init)
    s = tf_compat.v1.get_variable(
      "beta_" + name, shape=shape,
      initializer=tf.constant_initializer(beta_init)) if with_beta else None
    y = normalized_input * g
    if with_beta:
      y += s
    return y

  def _linear(self, inputs, out_dim, apply_bias=True, add_forget_bias=False, name=None):
    """
    :param tf.Tensor inputs: (B,D), or (T,B,D)
    :param int out_dim: transformed inputs dimension
    :param bool apply_bias: if True, then add bias to transformed inputs
    :param bool add_forget_bias: if True, then forget bias is added for forget gates
    :param str name: weight variable scope name
    :return: (B,out_dim) or (T,B,out_dim)
    :rtype: tf.Tensor
    """
    assert name is not None
    from returnn.tf.util.basic import dot
    input_dim = inputs.get_shape().dims[-1].value
    assert input_dim is not None, "%r shape unknown" % (inputs,)
    weights = tf_compat.v1.get_variable("W_" + name, shape=(input_dim, out_dim))
    out = dot(inputs, weights)
    if apply_bias:
      bias_init = [0.0] * out_dim
      if add_forget_bias and self.forget_bias > 0:
        assert 4 * self._num_units == out_dim
        bias_init[2*self._num_units:3*self._num_units] = [self.forget_bias] * self._num_units
      bias = tf_compat.v1.get_variable("bias_" + name, shape=[out_dim], initializer=tf.constant_initializer(bias_init))
      out += bias
    return out

  def _get_dropout_mask(self, dropout):
    """
    :param float dropout:
    :return: scalar (1.0) or shape (batch_size, num_units)
    :rtype: tf.Tensor
    """
    from returnn.tf.util.basic import default_control_flow_ctx, cond
    # Create the dropout masks outside the loop:
    with default_control_flow_ctx():
      def get_mask():
        """
        :rtype: tf.Tensor
        """
        batch_size = LayerBase.get_recent_layer().get_batch_dim()
        keep_prob = 1.0 - dropout
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += tf_compat.v1.random_uniform(
          (batch_size, self._num_units), seed=self.dropout_seed, dtype=tf.float32)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = tf.floor(random_tensor)
        return binary_tensor * (1.0 / keep_prob)

      return cond(self.is_training, get_mask, lambda: 1.0)

  def _optional_dropout(self, x, dropout):
    """
    :param tf.Tensor x: (B,D)
    :param float dropout:
    :return: x, or x with dropout, (B,D)
    :rtype: tf.Tensor
    """
    if not dropout:
      return x
    if self.is_training is False:
      return x
    x *= self._get_dropout_mask(dropout=dropout)
    x.set_shape((None, self._num_units))
    return x

  def get_input_transformed(self, inputs, batch_dim=None):
    """
    :param tf.Tensor inputs:
    :param tf.Tensor|None batch_dim:
    :rtype: tf.Tensor
    """
    if self.with_concat:  # concat inputs, prev_h
      assert not self.global_norm, "%s: global_norm and with_concat together not supported" % self
      return inputs
    inputs = self._linear(inputs,
                          4 * self._num_units,
                          apply_bias=not self.global_norm and not self.global_norm_joined and not self.per_gate_norm,
                          add_forget_bias=not self.per_gate_norm,
                          name='ff')
    if self.global_norm:
      # `global_norm_joined` will not be enabled so it is safe to add beta
      # `per_gate_norm` will not be enabled so it is safe to add forget_bias
      inputs = self._norm(inputs, add_forget_bias=True, name='input_below')
    return inputs

  def __call__(self, inputs, state, scope=None):
    """
    Run this RNN cell on inputs given a state.

    :param tf.Tensor inputs:
    :param rnn_cell.LSTMStateTuple state:
    :return: (LSTM output h, LSTM state (consisting of cell state c and output h)
    :rtype: (tf.Tensor, rnn_cell.LSTMStateTuple)
    """
    prev_c, prev_h = state
    prev_h = self._optional_dropout(prev_h, dropout=self.dropout_h)

    if self.with_concat:
      assert not self.global_norm
      concat_input = tf.concat([inputs, prev_h], axis=-1)
      lstm_in = self._linear(concat_input,
                             4 * self._num_units,
                             apply_bias=not self.per_gate_norm and not self.global_norm_joined,
                             add_forget_bias=True,
                             name='ff_re')
    else:
      # The input is already transformed by `get_input_transformed` function
      input_below = inputs
      # Bias already via get_input_transformed (if not global_norm, otherwise anyway should not been used).
      state_below = self._linear(prev_h, 4 * self._num_units, apply_bias=False, name='re')
      if self.global_norm:
        # Beta already in get_input_transformed.
        state_below = self._norm(state_below, name='state_below', with_beta=False)
      lstm_in = tf.add(input_below, state_below)
    if self.global_norm_joined:
      lstm_in = self._norm(lstm_in, add_forget_bias=True, name='lstm_in')

    i, j, f, o = tf.split(lstm_in, num_or_size_splits=4, axis=-1)

    if self.per_gate_norm:
      i = self._norm(i, name='i_gate')
      j = self._norm(j, name='j_gate')
      f = self._norm(f, add_forget_bias=True, name='f_gate')
      o = self._norm(o, name='o_gate')

    g = self._optional_dropout(self.activation(j), dropout=self.dropout)

    from tensorflow.python.ops.math_ops import sigmoid

    new_c = sigmoid(f) * prev_c + sigmoid(i) * g
    new_c_for_output = new_c
    if self.cell_norm:
      new_c = self._norm(new_c, name='new_c')
      if self.cell_norm_in_output:
        new_c_for_output = new_c

    new_h = sigmoid(o) * self.activation(new_c_for_output)
    if self.hidden_norm:
      new_h = self._norm(new_h, name='new_h')

    return new_h, rnn_cell.LSTMStateTuple(new_c, new_h)


class TwoDLSTMLayer(LayerBase):
  """
  2D LSTM.

  Currently only from left-to-right in the time axis.
  Can be inside a recurrent loop, or outside.
  """
  layer_class = "twod_lstm"
  recurrent = True

  def __init__(self,
               pooling='last',
               unit_opts=None,
               forward_weights_init=None, recurrent_weights_init=None, bias_init=None,
               **kwargs):
    """
    :param str pooling: defines how the 1D return value is computed based on the 2D lstm result. Either 'last' or 'max'
    :param None|dict[str] unit_opts: passed to RNNCell creation
    :param str forward_weights_init: see :func:`returnn.tf.util.basic.get_initializer`
    :param str recurrent_weights_init: see :func:`returnn.tf.util.basic.get_initializer`
    :param str bias_init: see :func:`returnn.tf.util.basic.get_initializer`
    """
    super(TwoDLSTMLayer, self).__init__(**kwargs)
    import re
    from returnn.tf.util.basic import is_gpu_available_in_session
    assert is_gpu_available_in_session(), "currently, there's no CPU support"
    self.pooling = pooling
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
    #   https://www.tensorflow.org/api_docs/python/tf/initializers
    #   https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Orthogonal
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
    from returnn.tf.util.basic import get_initializer, xavier_initializer
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
    with reuse_name_scope("rec-twod", initializer=default_var_initializer) as scope:
      assert isinstance(scope, tf_compat.v1.VariableScope)
      self._rec_scope = scope
      scope_name_prefix = scope.name + "/"  # e.g. "layer1/rec/"
      with self.var_creation_scope():
        self.cell = self._get_cell(unit_opts=unit_opts)

      # this must not be part of var_creation_scope - otherwise the used operations appear to TF to be used outside
      # of the while loop, leading to errors
      y = self._get_output_native_rec_op(self.cell)

      self.output.placeholder = y

      # Very generic way to collect all created params.
      # Note that for the TF RNN cells, there is no other way to do this.
      # Also, see the usage of :func:`LayerBase.cls_layer_scope`, e.g. for initial vars.
      params = tf_compat.v1.get_collection(tf_compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=re.escape(scope_name_prefix))
      self._add_params(params=params, scope_name_prefix=scope_name_prefix)

  @classmethod
  def get_out_data_from_opts(cls, sources, n_out, name, **kwargs):
    """
    :param list[LayerBase] sources:
    :param int n_out:
    :param str name:
    :rtype: Data
    """
    assert len(sources) == 2, "Exactly 2 sources (x and y axis) have to be specified."
    batch_dim_axis = sources[1].output.batch_dim_axis
    time_dim_axis = sources[1].output.time_dim_axis
    shape = sources[1].output.shape[:-1] + (n_out,)
    size_placeholder = sources[1].output.size_placeholder.copy()
    beam = sources[0].output.beam
    dtype = "float32"
    available_for_inference = all([src.output.available_for_inference for src in sources])

    return Data(
      name="%s_output" % name,
      shape=shape,
      batch_dim_axis=batch_dim_axis,
      time_dim_axis=time_dim_axis,
      size_placeholder=size_placeholder,
      available_for_inference=available_for_inference,
      dtype=dtype,
      beam=beam,
      sparse=False)

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

  def _get_input(self):
    """
    :return: (x, seq_len), where x is (time,batch,...,dim) and seq_len is (batch,)
    :rtype: (tf.Tensor, tf.Tensor)
    """
    assert len(self.sources) == 2
    assert self.sources[0].output
    assert self.sources[1].output
    x = self.sources[0].output.get_placeholder_as_time_major()  # (time,batch,[dim])
    seq_len_src = self.sources[0].output.get_sequence_lengths()

    return x, seq_len_src

  def get_constraints_value(self):
    """
    :rtype: tf.Tensor
    """
    v = super(TwoDLSTMLayer, self).get_constraints_value()
    from returnn.tf.util.basic import optional_add
    if isinstance(self.cell, _SubnetworkRecCell):
      for layer in self.cell.net.layers.values():
        v = optional_add(v, layer.get_constraints_value())
    return v

  def _get_cell(self, unit_opts=None):
    """
    :param None|dict[str] unit_opts:
    :rtype: returnn.tf.native_op.TwoDNativeLstmCell
    """
    import returnn.tf.native_op as tf_native_op
    rnn_cell_class = tf_native_op.TwoDNativeLstmCell
    n_hidden = self.output.dim
    if unit_opts is None:
      unit_opts = {}

    assert not self.sources[0].output.sparse
    n_input_dim_parts = [self.sources[0].output.dim, self.sources[1].output.dim]
    cell = rnn_cell_class(
      n_hidden=n_hidden, n_input_dim=sum(n_input_dim_parts), n_input_dim_parts=n_input_dim_parts,
      input_is_sparse=self.sources[0].output.sparse,
      pooling=self.pooling,
      **unit_opts)
    return cell

  @classmethod
  def helper_extra_outputs(cls, batch_dim, src_length, features):
    """
    :param tf.Tensor batch_dim:
    :param tf.Tensor src_length:
    :param tf.Tensor|int features:
    :rtype: dict[str,tf.Tensor]
    """
    return {"state": tf.zeros([batch_dim, 1, src_length, 5 * features]),
            "output": tf.zeros([batch_dim, 1, src_length, features]),
            "iteration": tf.zeros([batch_dim])}

  # noinspection PyMethodOverriding
  @classmethod
  def get_rec_initial_extra_outputs(cls, batch_dim, n_out, sources, **kwargs):
    """
    :param tf.Tensor batch_dim:
    :param int n_out:
    :param list[LayerBase] sources:
    :rtype: dict[str,tf.Tensor]
    """
    if sources[1].output.time_dim_axis is None:
      assert sources[0].output.time_dim_axis is not None
      src_length = tf.reduce_max(sources[0].output.get_sequence_lengths())
      return cls.helper_extra_outputs(batch_dim, src_length, n_out)
    else:
      return {}

  # noinspection PyMethodOverriding
  @classmethod
  def get_rec_initial_extra_outputs_shape_invariants(cls, rec_layer, n_out, sources, **kwargs):
    """
    :param returnn.tf.layers.rec.RecLayer|LayerBase|None rec_layer: for the scope
    :param int n_out:
    :param list[LayerBase] sources:
    :return: optional shapes for the tensors by get_rec_initial_extra_outputs
    :rtype: dict[str,tf.TensorShape]
    """
    if sources[1].output.time_dim_axis is None:
      batch_dim = None
      src_length = None

      return {"state": tf.TensorShape((batch_dim, 1, src_length, 5 * n_out)),
              "output": tf.TensorShape((batch_dim, 1, src_length, n_out)),
              "iteration": tf.TensorShape((batch_dim,))}
    else:
      return {}

  def _get_output_native_rec_op(self, cell):
    """
    :param TFNativeOp.RecSeqCellOp cell:
    :return: output of shape (time, batch, dim)
    :rtype: tf.Tensor
    """
    from returnn.tf.util.basic import dot, sequence_mask_time_major, to_int32_64, set_param_axes_split_info

    assert self.sources[0].output
    x, seq_len_src = self._get_input()
    if cell.does_input_projection:
      # The cell get's x as-is. It will internally does the matrix mult and add the bias.
      pass
    else:
      weights = tf_compat.v1.get_variable(
        name="W", shape=(self.sources[0].output.dim, cell.n_input_dim), dtype=tf.float32,
        initializer=self._fwd_weights_initializer)
      if self.sources[0].output.sparse:
        x = tf.nn.embedding_lookup(weights, to_int32_64(x))
      else:
        x = dot(x, weights)
      b = tf_compat.v1.get_variable(
        name="b", shape=(cell.n_input_dim,), dtype=tf.float32, initializer=self._bias_initializer)
      if len(cell.n_input_dim_parts) > 1:
        set_param_axes_split_info(weights, [[self.sources[0].output.dim], cell.n_input_dim_parts])
        set_param_axes_split_info(b, [cell.n_input_dim_parts])
      x += b
    index_src = sequence_mask_time_major(seq_len_src, maxlen=self.sources[0].output.time_dimension())

    # If the target does not have a time dimension, we have to add it
    if self.sources[1].output.time_dim_axis is None:
      targets = self.sources[1].output.get_placeholder_as_batch_major()  # (batch, trg_features)
      targets = tf.expand_dims(targets, 0)  # (1, batch, trg_features)
    else:
      targets = self.sources[1].output.get_placeholder_as_time_major()  # (trg_length, batch, trg_features)

    if self._rec_previous_layer:
      previous_state = self._rec_previous_layer.rec_vars_outputs["state"]  # (batch, 1, src_length, n_hidden)
      previous_output = self._rec_previous_layer.rec_vars_outputs["output"]  # (batch, 1, src_length, n_hidden)
      iteration = self._rec_previous_layer.rec_vars_outputs["iteration"]  # (batch,)
    else:
      batch_dim = tf.shape(targets)[1]
      sources = self.sources[0].output.get_placeholder_as_time_major()
      src_length = tf.shape(sources)[0]
      features = tf.shape(sources)[2]
      initial_values = TwoDLSTMLayer.helper_extra_outputs(batch_dim, src_length, features)

      previous_state = initial_values["state"]    # (batch, 1, src_length, n_hidden)
      previous_output = initial_values["output"]  # (batch, 1, src_length, n_hidden)
      iteration = initial_values["iteration"]     # (batch,)

    # to support the selection of the correct previous states and outputs, they have to be stored in batch mayor format
    # the c code needs them to be in time mayor (trg, src) format, so we have to swap the axes
    previous_state = tf.transpose(previous_state, perm=[1, 2, 0, 3])    # (1, src_length, batch, n_hidden)
    previous_output = tf.transpose(previous_output, perm=[1, 2, 0, 3])  # (1, src_length, batch, n_hidden)

    # noinspection PyTupleAssignmentBalance,PyArgumentList
    y, complete_output, final_state = cell(
      source=x, src_mask=index_src,
      recurrent_weights_initializer=self._rec_weights_initializer,
      target=targets,
      previous_state=previous_state,
      previous_output=previous_output,
      iteration=iteration)
    # y (trg_length, batch, n_hidden)
    # complete_out (trg_length, src_length, batch, n_hidden)
    # final_state (trg_length, src_length, batch, n_hidden*5)

    # swap axes again, to get back to the batch mayor format that's required by RETURNN
    final_state = tf.transpose(final_state, perm=[2, 0, 1, 3])          # (batch, trg_length, src_length, features)
    complete_output = tf.transpose(complete_output, perm=[2, 0, 1, 3])  # (batch, trg_length, src_length, features)

    final_state = final_state[:, -1:, :, :]          # (batch, 1, src_length, features)
    complete_output = complete_output[:, -1:, :, :]  # (batch, 1, src_length, features)

    self.rec_vars_outputs["state"] = final_state
    self.rec_vars_outputs["output"] = complete_output
    self.rec_vars_outputs["iteration"] = iteration + 1

    # during inference, the 2D result has target length 1. This dimension has to be removed to be conform with RETURNN
    if self.network.have_rec_step_info():
      y = y[0]

    return y


class ZoneoutLSTMCell(BaseRNNCell):
  """
  Wrapper for tf LSTM to create Zoneout LSTM Cell.
  This code is an adapted version of Rayhane Mamas version of Tacotron-2

  Refs:

    https://github.com/Rayhane-mamah/Tacotron-2
    https://arxiv.org/pdf/1606.01305.pdf
  """

  def __init__(self, num_units, zoneout_factor_cell=0., zoneout_factor_output=0.):
    """
    Initializer with possibility to set different zoneout values for cell/hidden states.

    :param int num_units: number of hidden units
    :param float zoneout_factor_cell: cell zoneout factor
    :param float zoneout_factor_output: output zoneout factor
    """
    super(ZoneoutLSTMCell, self).__init__()

    zm = min(zoneout_factor_output, zoneout_factor_cell)
    zs = max(zoneout_factor_output, zoneout_factor_cell)
    if zm < 0. or zs > 1.:
      raise ValueError('One/both provided Zoneout factors are not in [0, 1]')

    self._cell = rnn_cell.LSTMCell(num_units, state_is_tuple=True)
    self._zoneout_cell = zoneout_factor_cell
    self._zoneout_outputs = zoneout_factor_output
    from returnn.tf.network import TFNetwork
    self.is_training = TFNetwork.get_current_network().train_flag

  @property
  def state_size(self):
    """
    :rtype: int
    """
    return self._cell.state_size

  @property
  def output_size(self):
    """
    :rtype: int
    """
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    """
    Apply ZoneoutLSTM on input with given state

    :param tf.Tensor inputs: input tensor to the cell
    :param tf.nn.rnn_cell.LSTMStateTuple state: previous state of the LSTM
    :param tf.compat.v1.VariableScope scope: VariableScope for the created subgraph
    :return: tuple of output and LSTMStateTuple
    :rtype: (tf.Tensor, tf.nn.rnn_cell.LSTMStateTuple)
    """
    # Apply vanilla LSTM
    output, new_state = self._cell(inputs, state, scope)

    (prev_c, prev_h) = state
    (new_c, new_h) = new_state

    from returnn.tf.util.basic import cond
    c = cond(self.is_training,
             lambda: (1 - self._zoneout_cell) * tf_compat.v1.nn.dropout(
               new_c - prev_c,
               keep_prob=(1 - self._zoneout_cell)) + prev_c,
             lambda: (1 - self._zoneout_cell) * new_c + self._zoneout_cell * prev_c)

    h = cond(self.is_training,
             lambda: (1 - self._zoneout_outputs) * tf_compat.v1.nn.dropout(
               new_h - prev_h,
               keep_prob=(1 - self._zoneout_outputs)) + prev_h,
             lambda: (1 - self._zoneout_outputs) * new_h + self._zoneout_outputs * prev_h)

    new_state = rnn_cell.LSTMStateTuple(c, h)

    return output, new_state


class RelativePositionalEncodingLayer(_ConcatInputLayer):
  """
  Relative positioning term as introduced by Shaw et al., 2018

  Usually added to Self-Attention using key_shift.
  Parts of the code are adapted from Tensor2Tensor (https://github.com/tensorflow/tensor2tensor).

  Example usage::

      d[output + '_rel_pos'] = {"class": "relative_positional_encoding",
                                "from": [output + '_self_att_laynorm'],
                                "n_out": self.EncKeyTotalDim // self.AttNumHeads,
                                "forward_weights_init": self.ff_init}
      d[output + '_self_att_att'] = {"class": "self_attention",
                                     "num_heads": self.AttNumHeads,
                                     "total_key_dim": self.EncKeyTotalDim,
                                     "n_out": self.EncValueTotalDim, "from": [output + '_self_att_laynorm'],
                                     "attention_left_only": False, "attention_dropout": self.attention_dropout,
                                     "forward_weights_init": self.ff_init,
                                     "key_shift": output + '_rel_pos'}

  """
  layer_class = "relative_positional_encoding"
  recurrent = True

  def __init__(self, n_out, forward_weights_init="glorot_uniform", clipping=16, fixed=False, **kwargs):
    """
    :param int n_out: Feature dimension of encoding.
    :param int clipping: After which distance to fallback to the last encoding
    :param bool fixed: Uses sinusoid positional encoding instead of learned parameters
    :param str forward_weights_init: see :func:`returnn.tf.util.basic.get_initializer`
    """
    super(RelativePositionalEncodingLayer, self).__init__(**kwargs)
    from returnn.tf.util.basic import get_initializer

    if not self.input_data.have_time_axis():
      offset = self.network.get_rec_step_index()
      length = self.network.get_rec_step_index() + 1
      time_dim_tag = self.output.dim_tags[1]
      assert time_dim_tag.dimension is None, "%s: unexpected output time dim tag %r" % (self, time_dim_tag)
      # See CumConcatLayer for similar logic
      if time_dim_tag.dyn_size_ext is None:
        time_dim_tag.dyn_size_ext = Data(
          name="%s:size-inside" % self.name,
          dim_tags=[],  # scalar
          placeholder=length, dtype="int32",
          batch=self.output.batch, control_flow_ctx=self.network.get_control_flow_ctx())
      elif time_dim_tag.dyn_size_ext.placeholder is None:
        time_dim_tag.dyn_size_ext.placeholder = length
    else:
      offset = 0
      length = tf.shape(self.input_data.placeholder)[self.input_data.time_dim_axis]

    if fixed:
      from returnn.tf.util.basic import get_positional_encoding
      encoding_matrix = get_positional_encoding(
        length=2 * clipping + 1,
        num_channels=n_out)  # shape [2 * clipping + 1, n_out]
    else:
      fwd_weights_initializer = get_initializer(
        forward_weights_init, seed=self.network.random.randint(2 ** 31), eval_local_ns={"layer": self})
      with self.var_creation_scope():
        encoding_matrix = self.add_param(tf_compat.v1.get_variable(
          name="encoding_matrix", shape=(2 * clipping + 1, n_out), initializer=fwd_weights_initializer))
    # encoding_matrix has shape [2 * clipping + 1, n_out]

    range_vec = tf.range(length) - offset  # [length]

    if self.input_data.have_time_axis():
      range_mat = tf.reshape(tf.tile(range_vec, [length]), [length, length])
      distance_mat = range_mat - tf.transpose(range_mat)  # [length,length]
    else:
      distance_mat = tf.reshape(range_vec, [1, length])  # [1,length]
    distance_mat_clipped = tf.clip_by_value(distance_mat, -clipping, clipping)
    # Shift values to be >= 0. Each integer still uniquely identifies a relative
    # position difference.
    position_info_indices = distance_mat_clipped + clipping  # [length|1,length]

    encoding = tf.gather(encoding_matrix, position_info_indices)  # [length|1,length,n_out]

    self.output.placeholder = encoding

  @classmethod
  def get_out_data_from_opts(cls, name, sources, n_out, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param int n_out:
    :rtype: Data
    """
    data = get_concat_sources_data_template(sources, name="%s_output" % name)
    # The result will be without batch dim.
    feature_dim_tag = Dim(
      kind=Dim.Types.Feature, description="%s_rel_pos_enc_feat" % name, dimension=n_out, auto_generated=True)
    if data.have_time_axis():
      time_dim_tag = data.get_time_dim_tag()
      # TODO using same dim tag twice will not be supported at some future point...
      data = data.copy_template_new_dim_tags((time_dim_tag, time_dim_tag, feature_dim_tag))
    else:
      # length will be ``network.get_rec_step_index() + 1``.
      dummy_dim_tag = Dim(
        kind=Dim.Types.Spatial, description="%s_rel_pos_enc_dummy" % name, dimension=1, auto_generated=True)
      time_dim_tag = Dim(
        kind=Dim.Types.Spatial, description="%s_rel_pos_enc_time" % name, dimension=None, auto_generated=True)
      data = data.copy_template_new_dim_tags((dummy_dim_tag, time_dim_tag, feature_dim_tag))
    return data


class CumConcatLayer(_ConcatInputLayer):
  """
  Concatenates all previous frames of a time-axis.
  Like :class:`CumsumLayer` uses `sum`, this layer uses `concat`.

  This layer can be used as a base for auto-regressive self-attention.

  This layer expects to be inside a :class:`RecLayer`.

  Inside a rec loop (not optimized out),
  this will concatenate the current input
  to the previous accumulated inputs.
  For an input of shape `input_shape`,
  it will output a tensor of shape `[new_dim] + input_shape`.
  `new_dim` (``out_spatial_dim``) is a special dimension, usually of length `i`,
  where `i` is the current loop frame,
  i.e. the length increases in every loop frame.
  `new_dim` is specified by a separate own dim tag.
  For example, in the first frame,
  this will be of shape `[1] + input_shape`,
  in the second frame shape `[2] + input_shape`,
  and so on,
  and in the last frame shape `[T] + input_shape`.

  Outside the rec loop (optimized out),
  this layer expects an input with the time dim of the rec layer,
  and returns the input as-is,
  but replacing the time dim tag with the dim tag `new_dim`
  converted as outside the loop.

  Normally the optimization should not matter for the user,
  i.e. for the user, the logical behavior is always as being inside the rec loop.
  Outside the loop,
  the output represents a tensor of shape `[T, new_dim] + input_shape`,
  although we actually have another `new_dim` outside the loop,
  and `T` is not actually there,
  but we still have all the information,
  because the last frame has all information.
  This `new_dim` outside the loop stores all the dynamic seq lengths
  per frame of the loop, i.e. the dyn seq len are extended of shape [B,T] or [T]
  (unlike usually just [B]).
  This way following layers use different seq lengths of `new_dim` for different loop frames,
  just like if the `T` dim would actually exist.
  """
  layer_class = "cum_concat"
  recurrent = True  # order matters

  def __init__(self, out_spatial_dim, axis=None, **kwargs):
    """
    :param Dim out_spatial_dim:
    :param Dim|None axis: to operate over. only single_step_dim supported currently, assumes to be inside rec layer
    """
    super(CumConcatLayer, self).__init__(**kwargs)
    if axis:
      from returnn.tf.util.data import single_step_dim
      assert axis == single_step_dim, "%r only axis %r == %r supported currently" % (self, axis, single_step_dim)
    rec_layer = self.network.get_rec_parent_layer(inside_loop=False)
    assert rec_layer, "%r must be used inside a RecLayer" % self
    out_axis = self.output.get_axis_from_description(out_spatial_dim)
    new_dim_ = self.output.dim_tags[out_axis]
    assert new_dim_.control_flow_ctx == self.output.control_flow_ctx == self.network.get_control_flow_ctx()

    if not self.input_data.has_axis(rec_layer.time_dim_tag):  # inside loop
      current_data = self.input_data.copy_compatible_to(self.output, unbroadcast=False)
      current_frame = current_data.placeholder  # [B, 1, ..., D]
      last_frames = self._rec_previous_layer.rec_vars_outputs["state"]  # [B, t, ..., D]
      concat_frames = tf.concat([last_frames, current_frame], axis=out_axis)  # [B, t+1, ..., D]
      self.rec_vars_outputs["state"] = concat_frames
      self.output.placeholder = concat_frames

      if not new_dim_.dyn_size_ext or new_dim_.dyn_size_ext.placeholder is None:
        # Unbroadcasting to [B] is not needed because any layers operating on this
        # should be able to handle extended dyn sizes.
        # Clipping it to the max length for sequences in the loop which are already ended
        # (i.e. considering the end flag)
        # is also not needed because any calculations after the end are irrelevant.
        # Note: In case we have some initial state/output, this can be extended.
        dyn_size = self.network.get_rec_step_index() + 1  # scalar
        new_dim_.dyn_size_ext = Data(
          name="%s:cum-concat:size-inside" % self.name,
          dim_tags=[],  # scalar
          placeholder=dyn_size, dtype="int32",
          batch=self.output.batch, control_flow_ctx=self.network.get_control_flow_ctx())

    else:  # outside loop
      # If not inside a rec loop, this layer is a no-op on the tensor.
      self.output.placeholder = self.input_data.placeholder

      # However, we used new dim tags, which were already prepared.
      # We now must fill in the extended dynamic size information.
      if not new_dim_.dyn_size_ext:
        # This must match the logic above for inside the loop.
        # Note: In case we have some initial state/output, this can be extended.
        dyn_size = tf.range(tf.math.reduce_max(rec_layer.time_dim_tag.dyn_size)) + 1  # [T]
        new_dim_.dyn_size_ext = Data(
          name="%s:cum-concat:size-outside" % self.name,
          dim_tags=[rec_layer.time_dim_tag],
          placeholder=dyn_size, dtype="int32",
          batch=self.output.batch, control_flow_ctx=self.network.get_control_flow_ctx())

  @classmethod
  def get_out_data_from_opts(cls, name, network, sources, out_spatial_dim, **kwargs):
    """
    :param str name:
    :param returnn.tf.network.TFNetwork network:
    :param list[LayerBase] sources:
    :param Dim out_spatial_dim:
    :rtype: Data
    """
    input_data = get_concat_sources_data_template(sources, name="%s_output" % name)
    assert network.is_inside_rec_layer(inside_loop=False), "CumConcatLayer %r must be used inside a RecLayer" % name
    rec_time_dim = network.get_inside_rec_time_dim(inside_loop=False)
    assert rec_time_dim
    ctx = network.get_control_flow_ctx()
    assert ctx == input_data.control_flow_ctx
    new_dim_in_ctx = out_spatial_dim.get_for_batch_ctx(batch=input_data.batch, ctx=ctx)

    if not input_data.has_axis(rec_time_dim):  # inside loop
      assert ctx and ctx.is_loop() and ctx.loop_spatial_dim == rec_time_dim

      new_dim_in_ctx.dyn_size_ext = Data(
        name="%s:cum-concat:size-inside" % name,
        dim_tags=[],  # scalar
        dtype="int32",
        batch=input_data.batch, control_flow_ctx=ctx)

      # Currently SelectSearchSourcesLayer assumes that all rec_vars_outputs are batch-major.
      # Therefore we here copy the input as batch-major, and then add the time axis at axis 1.
      # In the future, when SelectSearchSourcesLayer has support for this, we can change this to operate on axis 0,
      # which should be more efficient
      out = input_data.copy_as_batch_major()
      out = out.copy_add_dim_by_tag(new_dim_in_ctx, unbroadcast=True, axis=1)
      return out

    else:  # outside loop
      # Assume that the input has the time dim from the rec layer.
      axis = input_data.get_axis_from_description(rec_time_dim)
      return input_data.copy_template_replace_dim_tag(axis=axis, new_dim_tag=new_dim_in_ctx)

  # noinspection PyMethodOverriding
  @classmethod
  def get_rec_initial_extra_outputs(cls, network, batch_dim, rec_layer, sources, output, out_spatial_dim, **kwargs):
    """
    :param returnn.tf.network.TFNetwork network:
    :param tf.Tensor batch_dim:
    :param returnn.tf.layers.rec.RecLayer|LayerBase rec_layer:
    :param list[LayerBase] sources:
    :param Data output:
    :param Dim out_spatial_dim:
    :rtype: dict[str,tf.Tensor]
    """
    if network.is_inside_rec_layer():
      shape = []
      for tag in output.dim_tags:
        if tag.is_batch_dim():
          shape.append(batch_dim)
        elif tag == out_spatial_dim:
          shape.append(0)
        elif tag.dimension is not None:
          shape.append(tag.dimension)
        else:
          assert tag.dyn_size is not None
          shape.append(tf.math.reduce_max(tag.dyn_size))
      return {"state": tf.zeros(shape, dtype=output.dtype)}
    else:
      return {}

  # noinspection PyMethodOverriding
  @classmethod
  def get_rec_initial_extra_outputs_shape_invariants(cls, rec_layer, network, sources, output, **kwargs):
    """
    :param returnn.tf.layers.rec.RecLayer|LayerBase|None rec_layer: for the scope
    :param returnn.tf.network.TFNetwork network:
    :param list[LayerBase] sources:
    :param Data output:
    :rtype: dict[str, tf.TensorShape]
    """
    if network.is_inside_rec_layer():
      return {"state": tf.TensorShape(output.batch_shape)}
    else:
      return {}
