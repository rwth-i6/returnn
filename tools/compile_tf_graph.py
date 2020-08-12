#!/usr/bin/env python3

"""
Construct/compile the computation graph, and optionally save it to some file.
There are various options/variations for what task and what conditions you can create the graph,
e.g. for training, forwarding, search, or also step-by-step execution over a recurrent layer.
"""

from __future__ import print_function

import typing
import os
import sys
import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_io

import _setup_returnn_env  # noqa
import returnn.__main__ as rnn
from returnn.log import log
from returnn.config import Config
import returnn.util.basic as util
from returnn.util.basic import NotSpecified
import returnn.tf.compat as tf_compat
from returnn.tf.util.basic import Data, CollectionKeys
from returnn.tf.network import TFNetwork
from returnn.tf.layers.basic import LayerBase, register_layer_class
from returnn.tf.layers.base import WrappedInternalLayer
# noinspection PyProtectedMember
from returnn.tf.layers.rec import RecLayer, _SubnetworkRecCell, ChoiceLayer


config = None  # type: typing.Optional[Config]


def init(config_filename, log_verbosity):
  """
  :param str config_filename: filename to config-file
  :param int log_verbosity:
  """
  rnn.init_better_exchook()
  rnn.init_thread_join_hack()
  print("Using config file %r." % config_filename)
  assert os.path.exists(config_filename)
  rnn.init_config(config_filename=config_filename, extra_updates={
    "use_tensorflow": True,
    "log": None,
    "log_verbosity": log_verbosity,
    "task": __file__,  # just extra info for the config
  })
  global config
  config = rnn.config
  rnn.init_log()
  print("Returnn compile-tf-graph starting up.", file=log.v1)
  rnn.returnn_greeting()
  rnn.init_backend_engine()
  assert util.BackendEngine.is_tensorflow_selected(), "this is only for TensorFlow"
  rnn.init_faulthandler()
  rnn.init_config_json_network()


def create_graph(train_flag, eval_flag, search_flag, net_dict):
  """
  :param bool train_flag:
  :param bool eval_flag:
  :param bool search_flag:
  :param dict[str,dict[str]] net_dict:
  :return: adds to the current graph, and then returns the network
  :rtype: returnn.tf.network.TFNetwork
  """
  print("Loading network, train flag %s, eval flag %s, search flag %s" % (train_flag, eval_flag, search_flag))
  from returnn.tf.engine import Engine
  from returnn.tf.network import TFNetwork
  network, updater = Engine.create_network(
    config=config, rnd_seed=1,
    train_flag=train_flag, eval_flag=eval_flag, search_flag=search_flag,
    net_dict=net_dict)
  assert isinstance(network, TFNetwork)
  return network


class RecStepByStepLayer(RecLayer):
  """
  Represents a single step of :class:`RecLayer`.
  The purpose is to execute a single step only.
  This also takes care of all needed state, and latent variables (via :class:`ChoiceLayer`).
  All the state is kept in variables, such that you can avoid feeding/fetching.

  E.g., if you want to implement beam search in an external application which uses the compiled graph,
  you would compile the graph with search_flag disabled (such that RETURNN does not do any related logic),
  enable this recurrent step-by-step compilation, and then do the following TF session runs:

  * Call the initializers of all the state variables, which also includes everything from the base (e.g. encoder),
    while feeding in the placeholders for the input features.
    This is "init_op" in the info json.
    All further session runs should not need any feed values. All needed state should be in state vars.
    This will init all state vars, except stochastic_var_*.
  * Maybe you want to tile the batch dim now, according to your beam size.
    This is "tile_batch" in the info json.
  * For each decoder step:
    * For each stochastic variable (both latent variables and observable variables),
      in the order as in "stochastic_var_order" in the info json:
      - This calculation depends on all state vars
        (except of stochastic_var_scores_* and only the dependent other stochastic_var_choice_*).
      - Calculate `"stochastic_var_scores_%s" % name` (which are the probabilities in +log space).
        This is "calc_scores_op" in the info json.
      - Do a choice, build up a beam of hypotheses.
      - Set the stochastic state var `"stochastic_var_choice_%s" % name` to the selected values (label indices).
      - If the beam has multiple items, i.e. the batch dimension changed, you must make sure
        that all further used state variables will also have the same batch dim.
      - You can use "select_src_beams" to select the new states given the choices.
    * Do a single session run for the next values of these state vars:
      "i", "end_flag" (if existing), "dyn_seq_len" (if existing), "state_*" (multiple vars).
      These are also the state vars which will get updated in every further recurrent step.
      The "base_*" state vars are always kept (although you might need to update the batch dim),
      and the "stochastic_var_*" state vars have the special logic for the stochastic variables.
      This is "next_step_op" in the info json.
  """
  layer_class = "rec_step_by_step"

  @classmethod
  def prepare_compile(cls, rec_layer_name, net_dict):
    """
    :param str rec_layer_name:
    :param dict[str,dict[str]] net_dict:
    :return: nothing, will prepare globally, and modify net_dict in place
    """
    register_layer_class(RecStepByStepLayer)
    register_layer_class(ChoiceStateVarLayer)
    assert rec_layer_name in net_dict
    rec_layer_dict = net_dict[rec_layer_name]
    assert rec_layer_dict["class"] == "rec"
    assert isinstance(rec_layer_dict["unit"], dict)
    rec_layer_dict["class"] = RecStepByStepLayer.layer_class

  @classmethod
  def post_compile(cls, rec_layer_name, network, output_file_name=None):
    """
    :param str rec_layer_name:
    :param TFNetwork network:
    :param str|None output_file_name:
    """
    assert rec_layer_name in network.layers
    rec_layer = network.layers[rec_layer_name]
    assert isinstance(rec_layer, RecStepByStepLayer)
    info = {"state_vars": {}, "stochastic_var_order": [], "stochastic_vars": {}}
    init_ops = []
    tile_batch_repetitions = tf_compat.v1.placeholder(name="tile_batch_repetitions", shape=(), dtype=tf.int32)
    tile_batch_ops = []
    src_beams = tf_compat.v1.placeholder(name="src_beams", shape=(None, None), dtype=tf.int32)  # (batch,beam)
    select_src_beams_ops = []
    next_step_ops = []
    print("State vars:")
    for name, var in sorted(rec_layer.state_vars.items()):
      assert isinstance(name, str)
      assert isinstance(var, RecStepByStepLayer.StateVar)
      print(" %s: %r, shape %s, dtype %s" % (name, var.var.op.name, var.var.shape, var.var.dtype.base_dtype.name))
      info["state_vars"][name] = {
        "var_op": var.var.op.name,
        "shape": [int(d) if d is not None else None for d in var.var_data_shape.batch_shape],
        "dtype": var.var.dtype.base_dtype.name}
      if not name.startswith("stochastic_var_"):
        init_ops.append(var.init_op())
        tile_batch_ops.append(var.tile_batch_op(tile_batch_repetitions))
        select_src_beams_ops.append(var.select_src_beams_op(src_beams=src_beams))
      if not name.startswith("stochastic_var_") and not name.startswith("base_"):
        next_step_ops.append(var.final_op())
    info["init_op"] = tf.group(*init_ops, name="rec_step_by_step_init_op").name
    info["next_step_op"] = tf.group(*next_step_ops, name="rec_step_by_step_update_op").name
    info["tile_batch"] = {
      "op": tf.group(*tile_batch_ops, name="rec_step_by_step_tile_batch_op").name,
      "repetitions_placeholder": tile_batch_repetitions.op.name}
    info["select_src_beams"] = {
      "op": tf.group(*select_src_beams_ops, name="rec_step_by_step_select_src_beams_op").name,
      "src_beams_placeholder": src_beams.op.name}
    print("Stochastic vars, and their order:")
    for name in rec_layer.stochastic_var_order:
      print(" %s" % name)
      info["stochastic_var_order"].append(name)
      info["stochastic_vars"][name] = {
        "calc_scores_op": rec_layer.state_vars["stochastic_var_scores_%s" % name].final_op().name,
        "scores_state_var": "stochastic_var_scores_%s" % name,
        "choice_state_var": "stochastic_var_choice_%s" % name}
    import json
    info_str = json.dumps(info, sort_keys=True, indent=2)
    print("JSON:")
    print(info_str)
    if not output_file_name:
      print("No rec-step-by-step output file name specified, not storing this info.")
    else:
      with open(output_file_name, "w") as f:
        f.write(info_str)
      print("Stored rec-step-by-step info JSON in file:", output_file_name)

  class StateVar:
    """
    Represents a state variable, i.e. either a state, a choice, or encoder state, etc.
    """

    def __init__(self, parent, name, initial_value, data_shape):
      """
      :param RecStepByStepLayer parent:
      :param str name:
      :param tf.Tensor|None initial_value:
        initial_value might have dim 1 in variable dimensions (which are not the batch-dim-axis),
        see get_rec_initial_output, which should be fine for broadcasting.
      :param Data data_shape:
        Describes the shape of initial_value, and also what we store as self.orig_data_shape,
        and what we return by self.read().
        If it is not a scalar, and batch-dim-axis > 0, the created variable will still be in batch-major.
      """
      self.parent = parent
      self.name = name
      self.orig_data_shape = data_shape
      self.var_data_shape = data_shape.copy_as_batch_major() if data_shape.batch_dim_axis is not None else data_shape
      del data_shape
      self.orig_initial_value = initial_value
      if initial_value is not None and self.orig_data_shape.batch_dim_axis not in (0, None):
        x = self.orig_data_shape.copy()
        x.placeholder = initial_value
        x = x.copy_compatible_to(self.var_data_shape)
        initial_value = x.placeholder
      self.var_initial_value = initial_value
      del initial_value
      # Note: Don't use `initializer` of `tf.get_variable` directly, because
      # it uses _try_guard_against_uninitialized_dependencies internally,
      # which replace references to variables in `initial_value` with references to the variable's initialized values.
      # This is not what we want. Also, it has a cycle check which is extremely inefficient and basically just hangs.
      # Instead, some dummy initializer. The shape should not matter.
      zero_initializer = tf.zeros(
        [d if (d is not None) else 1 for d in self.var_data_shape.batch_shape],
        dtype=self.var_data_shape.dtype)
      zero_initializer.set_shape(self.var_data_shape.batch_shape)
      self.var = tf_compat.v1.get_variable(
        name=name, initializer=zero_initializer, validate_shape=False)  # type: tf.Variable
      self.var.set_shape(self.var_data_shape.batch_shape)
      assert self.var.shape.as_list() == list(self.var_data_shape.batch_shape)
      print("New state var %r: %s, shape %s" % (name, self.var, self.var_data_shape))
      self.final_value = None  # type: typing.Optional[tf.Tensor]

    def __repr__(self):
      return "<StateVar %r, shape %r, initial %r>" % (self.name, self.var_data_shape, self.orig_initial_value)

    def set_final_value(self, final_value):
      """
      :param tf.Tensor final_value:
      """
      assert self.final_value is None
      assert isinstance(final_value, tf.Tensor)
      self.final_value = final_value

    def read(self):
      """
      :return: tensor in the format of self.orig_data_shape
      :rtype: tf.Tensor
      """
      value = self.var.read_value()
      if self.orig_data_shape.batch_dim_axis in (0, None):
        return value  # should be the right shape
      # Need to convert from self.var_data_shape to self.orig_data_shape (different batch-dim-axis).
      x = self.var_data_shape.copy()
      x.placeholder = value
      x = x.copy_compatible_to(self.orig_data_shape)
      return x.placeholder

    def init_op(self):
      """
      :return: op which assigns self.var_initial_value to self.var
      :rtype: tf.Operation
      """
      assert self.var_initial_value is not None
      return tf_compat.v1.assign(self.var, self.var_initial_value, name="init_state_var_%s" % self.name).op

    def final_op(self):
      """
      :return: op which assigns self.final_value (maybe converted) to self.var
      :rtype: tf.Operation
      """
      assert self.final_value is not None
      value = self.final_value
      from returnn.tf.util.basic import find_ops_path_output_to_input
      feed_tensors = []
      for data in self.parent.network.extern_data.data.values():
        feed_tensors.append(data.placeholder)
        feed_tensors.extend(data.size_placeholder.values())
      path = find_ops_path_output_to_input(fetches=value, tensors=feed_tensors)
      assert not path, "There should be no path from extern data to this final op value, but there is: %r" % (path,)
      if self.orig_data_shape.batch_dim_axis not in (0, None):
        x = self.orig_data_shape.copy()
        x.placeholder = value
        x = x.copy_compatible_to(self.var_data_shape)
        value = x.placeholder
      return tf_compat.v1.assign(self.var, value, name="final_state_var_%s" % self.name).op

    def tile_batch_op(self, repetitions):
      """
      :param tf.Tensor repetitions:
      :return: op which assigns the tiled value of the previous var value
      :rtype: tf.Operation
      """
      if self.var_data_shape.batch_dim_axis is None:
        return tf.no_op(name="tile_batch_state_var_no_op_%s" % self.name)
      # See also Data.copy_extend_with_beam.
      from returnn.tf.util.basic import tile_transposed
      tiled_value = tile_transposed(
        self.var.read_value(), axis=self.var_data_shape.batch_dim_axis, multiples=repetitions)
      return tf_compat.v1.assign(self.var, tiled_value, name="tile_batch_state_var_%s" % self.name).op

    def select_src_beams_op(self, src_beams):
      """
      :param tf.Tensor src_beams: (batch, beam) -> src-beam-idx
      :return: op which select the beams in the state var
      :rtype: tf.Operation
      """
      if self.var_data_shape.batch_dim_axis is None:
        return tf.no_op(name="select_src_beams_state_var_no_op_%s" % self.name)
      from returnn.tf.util.basic import select_src_beams
      v = select_src_beams(self.var.read_value(), src_beams=src_beams)
      return tf_compat.v1.assign(self.var, v, name="select_src_beams_state_var_%s" % self.name).op

  def __init__(self, **kwargs):
    kwargs = kwargs.copy()
    kwargs["optimize_move_layers_out"] = False
    sub_net_dict = kwargs["unit"]
    assert isinstance(sub_net_dict, dict)
    for key, layer_dict in list(sub_net_dict.items()):
      assert isinstance(layer_dict, dict)
      if layer_dict["class"] == "choice":
        layer_dict = layer_dict.copy()
        layer_dict["class"] = ChoiceStateVarLayer.layer_class
        sub_net_dict[key] = layer_dict
    kwargs["unit"] = sub_net_dict
    self.state_vars = {}  # type: typing.Dict[str,RecStepByStepLayer.StateVar]
    self.stochastic_var_order = []  # type: typing.List[str]
    super(RecStepByStepLayer, self).__init__(**kwargs)

  def create_state_var(self, name, initial_value=None, data_shape=None):
    """
    A state var is a variable where the initial value is given by the encoder, or a constant,
    and the final value is determined by one step of this rec layer (usually called the decoder).

    :param str name:
    :param tf.Tensor|None initial_value: assumes batch-major, if data_shape is not given
    :param Data|None data_shape:
    :rtype: tf.Tensor
    """
    assert name not in self.state_vars
    assert data_shape or initial_value is not None
    if data_shape:
      assert isinstance(data_shape, Data)
    elif initial_value.shape.ndims == 0:
      data_shape = Data(name=name, batch_dim_axis=None, shape=(), dtype=initial_value.dtype.name)
    else:
      assert initial_value.shape.dims[0].value is None  # first is batch dim
      data_shape = Data(
        name=name, batch_dim_axis=0, shape=initial_value.shape.as_list()[1:], dtype=initial_value.dtype.name)
    if initial_value is not None:
      # initial_value might have dim 1 in variable dimensions (which are not the batch-dim-axis),
      # see get_rec_initial_output, which should be fine for broadcasting.
      initial_value.set_shape(data_shape.batch_shape)
    var = self.StateVar(parent=self, name=name, initial_value=initial_value, data_shape=data_shape)
    self.state_vars[name] = var
    return var.read()

  def set_state_var_final_value(self, name, final_value):
    """
    :param str name:
    :param tf.Tensor final_value:
    """
    self.state_vars[name].set_final_value(final_value)

  def create_state_vars_recursive(self, name_prefix, initial_values, data_shape=None):
    """
    :param str|list[str] name_prefix:
    :param T initial_values:
    :param data_shape: same structure as initial_values, but values are of instance :class:`Data`
    :return: same as initial_values, but the variables
    :rtype: T
    """
    from returnn.util.basic import make_seq_of_type
    if isinstance(name_prefix, (tuple, list)):
      assert isinstance(initial_values, (tuple, list))
      assert len(name_prefix) == len(initial_values)
      if data_shape is not None:
        assert isinstance(data_shape, (tuple, list)) and len(data_shape) == len(initial_values)
      return make_seq_of_type(
        type(initial_values),
        [self.create_state_vars_recursive(
          name_prefix=name_prefix[i], initial_values=v, data_shape=data_shape[i] if data_shape else None)
         for i, v in enumerate(initial_values)])
    if initial_values is None:
      assert data_shape is None
      return None
    if isinstance(initial_values, tf.Tensor):
      return self.create_state_var(name=name_prefix, initial_value=initial_values, data_shape=data_shape)
    if isinstance(initial_values, (tuple, list)):
      if data_shape is not None:
        assert isinstance(data_shape, (tuple, list)) and len(data_shape) == len(initial_values)
      return make_seq_of_type(
        type(initial_values),
        [self.create_state_vars_recursive(
          name_prefix="%s_%i" % (name_prefix, i), initial_values=v, data_shape=data_shape[i] if data_shape else None)
         for i, v in enumerate(initial_values)])
    if isinstance(initial_values, dict):
      if data_shape is not None:
        assert isinstance(data_shape, dict) and set(data_shape.keys()) == set(initial_values.keys())
      return {
        k: self.create_state_vars_recursive(
          name_prefix="%s_%s" % (name_prefix, k), initial_values=v, data_shape=data_shape[k] if data_shape else None)
        for k, v in initial_values.items()}
    raise TypeError("unhandled type %r" % (initial_values,))

  def set_state_vars_final_values_recursive(self, name_prefix, final_values):
    """
    :param str|list[str] name_prefix:
    :param T final_values:
    :rtype: T
    """
    from returnn.util.basic import make_seq_of_type
    if isinstance(name_prefix, (tuple, list)):
      assert isinstance(final_values, (tuple, list))
      assert len(name_prefix) == len(final_values)
      return make_seq_of_type(
        type(final_values),
        [self.set_state_vars_final_values_recursive(name_prefix=name_prefix[i], final_values=v)
         for i, v in enumerate(final_values)])
    if final_values is None:
      return None
    if isinstance(final_values, tf.Tensor):
      return self.set_state_var_final_value(name=name_prefix, final_value=final_values)
    if isinstance(final_values, (tuple, list)):
      return make_seq_of_type(
        type(final_values),
        [self.set_state_vars_final_values_recursive(name_prefix="%s_%i" % (name_prefix, i), final_values=v)
         for i, v in enumerate(final_values)])
    if isinstance(final_values, dict):
      return {
        k: self.set_state_vars_final_values_recursive(name_prefix="%s_%s" % (name_prefix, k), final_values=v)
        for k, v in final_values.items()}
    raise TypeError("unhandled type %r" % (final_values,))

  def get_batch_dim_from_state(self):
    """
    :return: batch-dim, from some (any) state var, scalar, int32
    :rtype: tf.Tensor
    """
    for name, v in sorted(self.state_vars.items()):
      assert isinstance(v, RecStepByStepLayer.StateVar)
      if v.var_data_shape.batch_dim_axis is not None:
        with tf.name_scope("batch_dim_from_state_%s" % v.name):
          return tf.shape(v.var)[v.var_data_shape.batch_dim_axis]
    raise Exception("None of the state vars do have a batch-dim: %s" % self.state_vars)

  def add_stochastic_var(self, name):
    """
    :param str name:
    """
    assert name not in self.stochastic_var_order
    self.stochastic_var_order.append(name)

  def _get_cell(self, unit, unit_opts=None):
    """
    :param str|dict[str] unit:
    :param None|dict[str] unit_opts:
    :rtype: _SubnetworkRecCell
    """
    assert isinstance(unit, dict)
    assert unit_opts is None
    return SubnetworkRecCellSingleStep(parent_rec_layer=self, net_dict=unit)

  @classmethod
  def get_out_data_from_opts(cls, **kwargs):
    """
    :rtype: Data
    """
    return RecLayer.get_out_data_from_opts(**kwargs)


class ChoiceStateVarLayer(LayerBase):
  """
  Like :class:`ChoiceLayer`, but we don't do the search/choice ourselves,
  instead we store the scores in a variable, and the final result is another variable,
  which is expected to be set externally.
  This is expected to be used together with :class:`RecStepByStepLayer`.
  """
  layer_class = "choice_state_var"

  # noinspection PyUnusedLocal
  def __init__(self, beam_size,
               search=NotSpecified,
               input_type="prob",
               prob_scale=1.0, base_beam_score_scale=1.0, random_sample_scale=0.0,
               length_normalization=True,
               custom_score_combine=None,
               source_beam_sizes=None, scheduled_sampling=False, cheating=False,
               explicit_search_sources=None,
               **kwargs):
    super(ChoiceStateVarLayer, self).__init__(**kwargs)
    rec_layer = self.network.parent_layer
    assert isinstance(rec_layer, RecStepByStepLayer)
    assert len(self.sources) == 1
    source = self.sources[0]
    assert source.output.is_batch_major and len(source.output.shape) == 1
    scores_in = source.output.placeholder
    if input_type == "prob":
      if source.output_before_activation:
        scores_in = source.output_before_activation.get_log_output()
      else:
        from returnn.tf.util.basic import safe_log
        scores_in = safe_log(scores_in)
    elif input_type == "log_prob":
      pass
    else:
      raise ValueError("Not handled input_type %r" % (input_type,))
    rec_layer.create_state_var(
      name="stochastic_var_scores_%s" % self.name, data_shape=source.output)
    rec_layer.set_state_var_final_value(
      name="stochastic_var_scores_%s" % self.name, final_value=scores_in)
    self.output.placeholder = rec_layer.create_state_var(
      name="stochastic_var_choice_%s" % self.name, data_shape=self.output)
    rec_layer.add_stochastic_var(self.name)

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
    if d.get("target", NotSpecified) is not None:
      assert "target" in d, "%s: specify 'target' explicitly" % (cls.__name__,)
      if isinstance(d["target"], str):
        d["target"] = [d["target"]]
      assert isinstance(d["target"], list)
      assert len(d["target"]) == len(d["from"])
    if d.get("explicit_search_source"):
      assert "explicit_search_sources" not in d
      d["explicit_search_sources"] = [get_layer(d.pop("explicit_search_source"))]
    elif d.get("explicit_search_sources"):
      assert isinstance(d["explicit_search_sources"], (list, tuple))
      d["explicit_search_sources"] = [get_layer(name) for name in d["explicit_search_sources"]]
    super(ChoiceStateVarLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)

  get_out_data_from_opts = ChoiceLayer.get_out_data_from_opts


class SubnetworkRecCellSingleStep(_SubnetworkRecCell):
  """
  Adapts :class:`_SubnetworkRecCell` such that we execute only a single step.
  """

  def __init__(self, **kwargs):
    self._parent_layers = {}  # type: typing.Dict[str,WrappedInternalLayer]
    super(SubnetworkRecCellSingleStep, self).__init__(**kwargs)

  def _get_parent_layer(self, layer_name):
    """
    :param str layer_name: without "base:" prefix
    :rtype: LayerBase
    """
    if layer_name in self._parent_layers:
      return self._parent_layers[layer_name]
    layer = self.parent_net.get_layer(layer_name)
    rec_layer = self.parent_rec_layer
    if rec_layer is None:  # at template construction
      return layer
    assert isinstance(rec_layer, RecStepByStepLayer)
    output = layer.output.copy()
    output.placeholder = rec_layer.create_state_var(
      name="base_value_%s" % layer_name, initial_value=output.placeholder, data_shape=output)
    from returnn.tf.util.basic import DimensionTag
    for i, size in list(output.size_placeholder.items()):
      dim_tag = DimensionTag.get_tag_from_size_tensor(size)
      if not dim_tag:
        print("Warning, no defined dim tag on %r, axis %i" % (layer, output.get_batch_axis(i)), file=log.v2)
        dim_tag = output.get_dim_tag(output.get_batch_axis(i))
        dim_tag.set_tag_on_size_tensor(size)
      new_size = rec_layer.create_state_var(name="base_size%i_%s" % (i, layer_name), initial_value=size)
      dim_tag.set_tag_on_size_tensor(new_size)
      output.size_placeholder[i] = new_size
    layer = WrappedInternalLayer(name=layer_name, network=self.parent_net, output=output, base_layer=layer)
    self._parent_layers[layer_name] = layer
    return layer

  def _while_loop(self, cond, body, loop_vars, shape_invariants):
    """
    :param function cond:
    :param function body:
    :param T loop_vars:
    :param S shape_invariants:
    :rtype: T

    def body(i, net_vars, acc_tas, seq_len_info=None)
      tf.Tensor i: loop counter, scalar
      net_vars: the accumulator values. see also self.get_init_loop_vars()
      list[tf.TensorArray] acc_tas: the output accumulator TensorArray
      (tf.Tensor,tf.Tensor)|None seq_len_info: tuple (end_flag, seq_len)
      return: [i + 1, a_flat, tas]: the updated counter + new accumulator values + updated TensorArrays
        rtype (tf.Tensor, object, list[tf.TensorArray])

    def cond(i, net_vars, acc_tas, seq_len_info=None)
      ...
      return: tf.Tensor bool, True or False
    """
    rec_layer = self.parent_rec_layer
    assert isinstance(rec_layer, RecStepByStepLayer)

    if len(loop_vars) == 3:
      i, net_vars, acc_tas = loop_vars
      seq_len_info = None
    else:
      i, net_vars, acc_tas, seq_len_info = loop_vars
      seq_len_info = rec_layer.create_state_vars_recursive(["end_flag", "dyn_seq_len"], seq_len_info)
    i = rec_layer.create_state_var("i", i)
    with tf.name_scope("state"):
      # See _SubnetworkRecCell.GetOutput.body.
      # net_vars is (prev_outputs_flat, prev_extra_flat), and prev_outputs_flat corresponds to self._initial_outputs,
      # where the keys are the layer names.
      # prev_extra is always expected to be batch-major.
      prev_outputs_data = [self.layer_data_templates[k].output for k in sorted(self._initial_outputs.keys())]
      net_vars = rec_layer.create_state_vars_recursive(
        name_prefix="state", initial_values=net_vars, data_shape=(prev_outputs_data, None))
    # We are ignoring acc_tas (the tensor arrays).

    # Some layers make explicit use of the (global data) batch-dim,
    # which they can get via TFNetwork.get_data_batch_dim().
    # This will add a dependency on the external data, which we want to avoid.
    # We can avoid that by taking the batch dim instead from one of the other states.
    # Note that this would be wrong in beam search.
    self.net._batch_dim = rec_layer.get_batch_dim_from_state()

    with tf.name_scope("cond"):
      rec_layer.create_state_var("cond", tf.constant(True))
      res = cond(i, net_vars, acc_tas, seq_len_info)
      rec_layer.set_state_var_final_value("cond", res)

    with tf.name_scope("body"):
      res = body(i, net_vars, acc_tas, seq_len_info)
      assert len(res) == len(loop_vars)
      if len(res) == 3:
        i, net_vars, acc_tas = res
        seq_len_info = None
      else:
        i, net_vars, acc_tas, seq_len_info = res
    if seq_len_info:
      rec_layer.set_state_vars_final_values_recursive(["end_flag", "dyn_seq_len"], seq_len_info)
    rec_layer.set_state_var_final_value("i", i)
    rec_layer.set_state_vars_final_values_recursive("state", net_vars)
    return res


def main(argv):
  """
  Main entry.
  """
  argparser = argparse.ArgumentParser(description='Compile some op')
  argparser.add_argument('config', help="filename to config-file")
  argparser.add_argument('--train', type=int, default=0, help='0 disable (default), 1 enable, -1 dynamic')
  argparser.add_argument('--eval', type=int, default=0, help='calculate losses. 0 disable (default), 1 enable')
  argparser.add_argument('--search', type=int, default=0, help='beam search. 0 disable (default), 1 enable')
  argparser.add_argument("--verbosity", default=4, type=int, help="5 for all seqs (default: 4)")
  argparser.add_argument("--summaries_tensor_name", help="create Tensor for tf.compat.v1.summary.merge_all()")
  argparser.add_argument("--rec_step_by_step", help="make step-by-step graph for this rec layer (eg. 'output')")
  argparser.add_argument("--rec_step_by_step_output_file", help="store meta info for rec_step_by_step (JSON)")
  argparser.add_argument("--output_file", help='allowed extensions: pb, pbtxt, meta, metatxt, logdir')
  argparser.add_argument("--output_file_model_params_list", help="line-based, names of model params")
  argparser.add_argument("--output_file_state_vars_list", help="line-based, name of state vars")
  args = argparser.parse_args(argv[1:])
  assert args.train in [0, 1, -1] and args.eval in [0, 1] and args.search in [0, 1]
  init(config_filename=args.config, log_verbosity=args.verbosity)
  assert 'network' in config.typed_dict
  net_dict = config.typed_dict["network"]
  if args.rec_step_by_step:
    RecStepByStepLayer.prepare_compile(rec_layer_name=args.rec_step_by_step, net_dict=net_dict)
  with tf.Graph().as_default() as graph:
    assert isinstance(graph, tf.Graph)
    print("Create graph...")
    # See :func:`Engine._init_network`.
    tf_compat.v1.set_random_seed(42)
    if args.train < 0:
      from returnn.tf.util.basic import get_global_train_flag_placeholder
      train_flag = get_global_train_flag_placeholder()
    else:
      train_flag = bool(args.train)
    eval_flag = bool(args.eval)
    search_flag = bool(args.search)
    network = create_graph(train_flag=train_flag, eval_flag=eval_flag, search_flag=search_flag, net_dict=net_dict)

    if args.rec_step_by_step:
      RecStepByStepLayer.post_compile(
        rec_layer_name=args.rec_step_by_step, network=network, output_file_name=args.rec_step_by_step_output_file)

    from returnn.tf.layers.base import LayerBase
    for layer in network.layers.values():
      assert isinstance(layer, LayerBase)
      if layer.output.time_dim_axis is None:
        continue
      if layer.output.batch_dim_axis is None:
        continue
      with layer.cls_layer_scope(layer.name):
        tf.identity(layer.output.get_placeholder_as_batch_major(), name="output_batch_major")

    tf.group(*network.get_post_control_dependencies(), name="post_control_dependencies")

    # Do some cleanup of collections which do not contain tensors or operations,
    # because the tf.train.import_meta_graph code might fail otherwise.
    tf_compat.v1.get_collection_ref(CollectionKeys.RETURNN_LAYERS).clear()

    if args.summaries_tensor_name:
      summaries_tensor = tf_compat.v1.summary.merge_all()
      assert isinstance(summaries_tensor, tf.Tensor), "no summaries in the graph?"
      tf.identity(summaries_tensor, name=args.summaries_tensor_name)

    if args.output_file and os.path.splitext(args.output_file)[1] in [".meta", ".metatxt"]:
      # https://www.tensorflow.org/api_guides/python/meta_graph
      saver = tf_compat.v1.train.Saver(
        var_list=network.get_saveable_params_list(), max_to_keep=2 ** 31 - 1)
      graph_def = saver.export_meta_graph()
    else:
      graph_def = graph.as_graph_def(add_shapes=True)

    print("Graph collection keys:", graph.get_all_collection_keys())
    print("Graph num operations:", len(graph.get_operations()))
    print("Graph def size:", util.human_bytes_size(graph_def.ByteSize()))

    if args.output_file:
      filename = args.output_file
      _, ext = os.path.splitext(filename)
      if ext == ".logdir":
        print("Write TF events to logdir:", filename)
        writer = tf_compat.v1.summary.FileWriter(logdir=filename)
        writer.add_graph(graph)
        writer.flush()
      else:
        assert ext in [".pb", ".pbtxt", ".meta", ".metatxt"], 'filename %r extension invalid' % filename
        print("Write graph to file:", filename)
        graph_io.write_graph(
          graph_def,
          logdir=os.path.dirname(filename),
          name=os.path.basename(filename),
          as_text=ext.endswith("txt"))
    else:
      print("Use --output_file if you want to store the graph.")

    if args.output_file_model_params_list:
      print("Write model param list to:", args.output_file_model_params_list)
      with open(args.output_file_model_params_list, "w") as f:
        for param in network.get_params_list():
          assert param.name[-2:] == ":0"
          f.write("%s\n" % param.name[:-2])

    if args.output_file_state_vars_list:
      print("Write state var list to:", args.output_file_state_vars_list)
      with open(args.output_file_state_vars_list, "w") as f:
        for param in tf_compat.v1.get_collection(CollectionKeys.STATE_VARS):
          assert param.name[-2:] == ":0"
          f.write("%s\n" % param.name[:-2])


if __name__ == '__main__':
  main(sys.argv)
