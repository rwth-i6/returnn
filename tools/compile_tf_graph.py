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
import contextlib
import tensorflow as tf
from tensorflow.python.framework import graph_io

my_dir = os.path.dirname(os.path.abspath(__file__))
returnn_dir = os.path.dirname(my_dir)
sys.path.insert(0, returnn_dir)

import rnn
from Log import log
from Config import Config
import argparse
import Util
from Util import NotSpecified
from TFUtil import Data, CollectionKeys
from TFNetwork import TFNetwork
from TFNetworkLayer import LayerBase, register_layer_class, WrappedInternalLayer
from TFNetworkRecLayer import RecLayer, _SubnetworkRecCell, ChoiceLayer


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
  assert Util.BackendEngine.is_tensorflow_selected(), "this is only for TensorFlow"
  rnn.init_faulthandler()
  rnn.init_config_json_network()


def create_graph(train_flag, eval_flag, search_flag, net_dict):
  """
  :param bool train_flag:
  :param bool eval_flag:
  :param bool search_flag:
  :param dict[str,dict[str]] net_dict:
  :return: adds to the current graph, and then returns the network
  :rtype: TFNetwork.TFNetwork
  """
  print("Loading network, train flag %s, eval flag %s, search flag %s" % (train_flag, eval_flag, search_flag))
  from TFEngine import Engine
  from TFNetwork import TFNetwork
  network, updater = Engine.create_network(
    config=config, rnd_seed=1,
    train_flag=train_flag, eval_flag=eval_flag, search_flag=search_flag,
    net_dict=net_dict)
  assert isinstance(network, TFNetwork)
  return network


@contextlib.contextmanager
def helper_variable_scope():
  """
  :rtype: tf.VariableScope
  """
  from TFUtil import reuse_name_scope
  with reuse_name_scope("IO", absolute=True) as scope:
    yield scope


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
      This is "update_op" in the info json.
  """
  layer_class = "rec_step_by_step"

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
    cell = rec_layer.cell
    assert isinstance(cell, SubnetworkRecCellSingleStep)

    # tmp structures
    init_base_ops_list = []
    init_ops_list = []
    update_ops_list = []
    post_update_ops_list = []

    # specific collections needed by sprint
    update_ops_coll = tf.get_collection_ref("update_ops")
    post_update_ops_coll = tf.get_collection_ref("post_update_ops")
    encode_ops_coll = tf.get_collection_ref("encode_ops")
    decode_ops_coll = tf.get_collection_ref("decode_ops")
    decoder_input_vars_coll = tf.get_collection_ref("decoder_input_vars")
    decoder_output_vars_coll = tf.get_collection_ref("decoder_output_vars")
    global_vars_coll = tf.get_collection_ref("global_vars")

    # json info
    info = { "state_vars": {}, "stochastic_var_order": [], "stochastic_vars": {},
             "collections": { "encode_ops": [], "update_ops": [], "post_update_ops": [],
               "decode_ops": [], "decoder_input_vars": [], "decoder_output_vars": [],
               "global_vars": [] } }

    # base vars
    print("State vars:")
    for name, var in sorted(rec_layer.state_vars.items()):
      assert isinstance(name, str)
      assert isinstance(var, RecStepByStepLayer.StateVar)
      print(" %s: %r, shape %s, dtype %s" % (name, var.var.op.name, var.var.shape, var.var.dtype.base_dtype.name))
      info["state_vars"][name] = {
        "var_op": var.var.op.name,
        "shape": [int(d) if d is not None else None for d in var.var_data_shape.batch_shape],
        "dtype": var.var.dtype.base_dtype.name}
      assert var.var not in network.get_saveable_params_list(), 'StateVars are not restorable from the original model check point'
      if name.startswith("base_"):
        init_base_ops_list.append(var.init_op())
        info["collections"]["encode_ops"].append(var.init_op().name)

    # tile batch and group base ops
    init_base_ops_list.append( cell.parent_tile_multiples_var.initializer )
    info["collections"]["encode_ops"].append(cell.parent_tile_multiples_var.initializer.name)
    init_base_op = tf.group(*init_base_ops_list, name="rec_step_by_step_init_base_op")

    # loop vars
    for name, var in sorted(rec_layer.state_vars.items()):
      if name.startswith("stochastic_var_") or name.startswith("base_"): continue
      # global vars (hypothesis independent)
      if name.startswith("global_"):
        global_vars_coll.append(var.var)
        info["collections"]["global_vars"].append(var.name)
      # dependency on init_base_op for each init_op
      with tf.control_dependencies([init_base_op]):  
        init_ops_list.append(var.init_op())
      info["collections"]["encode_ops"].append(var.init_op().name)
      # choice-dependent loop vars are not I/O stored, but need to be updated after feeding choices
      if var.choice_dependent:
        update_ops_list.append(var.final_op())
        info["collections"]["update_ops"].append(var.final_op().name)
      # update loop vars after the last decode op
      elif var.choice_dependent is False:
        post_update_ops_list.append(var.final_op())
        info["collections"]["post_update_ops"].append(var.final_op().name)

    # group encode_ops
    with tf.control_dependencies( [init_base_op] ):
      init_op = tf.group(*init_ops_list, name="rec_step_by_step_init_op")
    encode_ops_coll.append( init_op )
 
    # group post_update_ops
    post_update_op = tf.group(*post_update_ops_list, name="rec_step_by_step_post_update_op")
    post_update_ops_coll.append( post_update_op )

    # stochastic vars
    print("Stochastic vars, and their order:")
    tile_batch_multiple = None
    for name in rec_layer.stochastic_var_order:
      print(" %s" % name)
      info["stochastic_var_order"].append(name)
      info["stochastic_vars"][name] = {
        "calc_scores_op": rec_layer.state_vars["stochastic_var_scores_%s" % name].final_op().name,
        "scores_state_var": "stochastic_var_scores_%s" % name,
        "choice_state_var": "stochastic_var_choice_%s" % name}
      # collect decode_input_vars
      input_var = rec_layer.state_vars["stochastic_var_choice_%s" % name].var
      decoder_input_vars_coll.append(input_var)
      info["collections"]["decoder_input_vars"].append(input_var.name)
      # collect decode_output_vars
      output_var = rec_layer.state_vars["stochastic_var_scores_%s" % name].var
      decoder_output_vars_coll.append(output_var)
      info["collections"]["decoder_output_vars"].append(output_var.name)
      # automatic tile encodings w.r.t. choices
      if tile_batch_multiple is None:
        tile_batch_multiple = rec_layer.state_vars["stochastic_var_choice_%s" % name].read_batch_dim()
      # collect decode_ops
      decode_op = rec_layer.state_vars["stochastic_var_scores_%s" % name].final_op()
      decode_ops_coll.append(decode_op)
      info["collections"]["decode_ops"].append(decode_op.name)

    # automatic tile encodings w.r.t. choices
    tile_op = cell.set_parent_tile_multiple_op(tile_batch_multiple)
    update_ops_list.append(tile_op)
    info["collections"]["update_ops"].append(tile_op.name)
    
    # group update ops
    update_op = tf.group(*update_ops_list, name="rec_step_by_step_update_op")
    update_ops_coll.append( update_op )

    import json
    info_str = json.dumps(info, sort_keys=True, indent=2)
    if not output_file_name:
      print("No rec-step-by-step output file name specified, not storing this info.")
      print("JSON:")
      print(info_str)
    else:
      with open(output_file_name, "w") as f:
        f.write(info_str)
      print("Stored rec-step-by-step info JSON in file:", output_file_name)

  class StateVar:
    """
    Represents a state variable, i.e. either a state, a choice, or encoder state, etc.
    """

    def __init__(self, parent, name, initial_value, data_shape, choice_dependent):
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
      :param bool|None choice_dependent: choice dependent or not relevant
      """
      self.parent = parent
      self.name = name
      self.choice_dependent = choice_dependent
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
      # (state) variable collections #
      var_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
      if choice_dependent is False:
        var_collections.append(CollectionKeys.STATE_VARS)
      with helper_variable_scope():
        self.var = tf.get_variable(name=name, initializer=zero_initializer, validate_shape=False, collections=var_collections)  # type: tf.Variable
      self.var.set_shape(self.var_data_shape.batch_shape)
      assert self.var.shape.as_list() == list(self.var_data_shape.batch_shape)
      print("New state var %r: %s, shape %s" % (name, self.var, self.var_data_shape))
      self.final_value = None  # type: typing.Optional[tf.Tensor]

    def __repr__(self):
      return "<StateVar %r, shape %r, initial %r>" % (self.name, self.var_data_shape, self.orig_initial_value)

    def read_batch_dim(self):
      """
      :return: scalar, int32, the batch dim (including any beam)
      :rtype: tf.Tensor
      """
      value = self.var.read_value()
      assert self.var_data_shape.batch_dim_axis is not None
      return tf.shape(value)[self.var_data_shape.batch_dim_axis]

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
      return tf.assign(self.var, self.var_initial_value, name="init_state_var_%s" % self.name, validate_shape=False).op

    def final_op(self):
      """
      :return: op which assigns self.final_value (maybe converted) to self.var
      :rtype: tf.Operation
      """
      assert self.final_value is not None
      value = self.final_value
      from TFUtil import find_ops_path_output_to_input
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
      return tf.assign(self.var, value, name="final_state_var_%s" % self.name, validate_shape=False).op

    def tile_batch_op(self, repetitions):
      """
      :param tf.Tensor repetitions:
      :return: op which assigns the tiled value of the previous var value
      :rtype: tf.Operation
      """
      if self.var_data_shape.batch_dim_axis is None:
        return tf.no_op(name="tile_batch_state_var_no_op_%s" % self.name)
      # See also Data.copy_extend_with_beam.
      from TFUtil import tile_transposed
      tiled_value = tile_transposed(
        self.var.read_value(), axis=self.var_data_shape.batch_dim_axis, multiples=repetitions)
      return tf.assign(self.var, tiled_value, name="tile_batch_state_var_%s" % self.name).op

    def select_src_beams_op(self, src_beams):
      """
      :param tf.Tensor src_beams: (batch, beam) -> src-beam-idx
      :return: op which select the beams in the state var
      :rtype: tf.Operation
      """
      if self.var_data_shape.batch_dim_axis is None:
        return tf.no_op(name="select_src_beams_state_var_no_op_%s" % self.name)
      from TFUtil import select_src_beams
      v = select_src_beams(self.var.read_value(), src_beams=src_beams)
      return tf.assign(self.var, v, name="select_src_beams_state_var_%s" % self.name).op

  def create_state_var(self, name, initial_value=None, data_shape=None, choice_dependent=None):
    """
    A state var is a variable where the initial value is given by the encoder, or a constant,
    and the final value is determined by one step of this rec layer (usually called the decoder).

    :param str name:
    :param tf.Tensor|None initial_value: assumes batch-major, if data_shape is not given
    :param Data|None data_shape:
    :param bool|None:
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
    var = self.StateVar(parent=self, name=name, initial_value=initial_value, data_shape=data_shape, choice_dependent=choice_dependent)
    self.state_vars[name] = var
    return var.read()

  def set_state_var_final_value(self, name, final_value):
    """
    :param str name:
    :param tf.Tensor final_value:
    """
    self.state_vars[name].set_final_value(final_value)

  def create_state_vars_recursive(self, name_prefix, initial_values, data_shape=None, choice_dependent=None):
    """
    :param str|list[str] name_prefix:
    :param T initial_values:
    :param data_shape: same structure as initial_values, but values are of instance :class:`Data`
    :param choice_dependent: same structure as initial_values, but values are of bool, or None
    :return: same as initial_values, but the variables
    :rtype: T
    """
    from Util import make_seq_of_type
    if isinstance(name_prefix, (tuple, list)):
      assert isinstance(initial_values, (tuple, list))
      assert len(name_prefix) == len(initial_values)
      if data_shape is not None:
        assert isinstance(data_shape, (tuple, list)) and len(data_shape) == len(initial_values)
      return make_seq_of_type(
        type(initial_values),
        [self.create_state_vars_recursive(
          name_prefix=name_prefix[i], initial_values=v, 
          data_shape=data_shape[i] if data_shape else None,
          choice_dependent=choice_dependent[i] if choice_dependent else None )
         for i, v in enumerate(initial_values)])
    if initial_values is None:
      assert data_shape is None and choice_dependent is None
      return None
    if isinstance(initial_values, tf.Tensor):
      return self.create_state_var(name=name_prefix, initial_value=initial_values, data_shape=data_shape, choice_dependent=choice_dependent)
    if isinstance(initial_values, (tuple, list)):
      if data_shape is not None:
        assert isinstance(data_shape, (tuple, list)) and len(data_shape) == len(initial_values)
      if choice_dependent is not None:
        assert isinstance(choice_dependent, (tuple, list)) and len(choice_dependent) == len(initial_values)
      return make_seq_of_type(
        type(initial_values),
        [self.create_state_vars_recursive(
          name_prefix="%s_%i" % (name_prefix, i), initial_values=v, 
          data_shape=data_shape[i] if data_shape else None,
          choice_dependent=choice_dependent[i] if choice_dependent else None )
         for i, v in enumerate(initial_values)])
    if isinstance(initial_values, dict):
      if data_shape is not None:
        assert isinstance(data_shape, dict) and set(data_shape.keys()) == set(initial_values.keys())
      if choice_dependent is not None:
        assert isinstance(choice_dependent, dict) and set(choice_dependent.keys()) == set(initial_values.keys())
      return {
        k: self.create_state_vars_recursive(
          name_prefix="%s_%s" % (name_prefix, k), initial_values=v, 
          data_shape=data_shape[k] if data_shape else None,
          choice_dependent=choice_dependent[k] if choice_dependent else None )
        for k, v in initial_values.items()}
    raise TypeError("unhandled type %r" % (initial_values,))

  def set_state_vars_final_values_recursive(self, name_prefix, final_values):
    """
    :param str|list[str] name_prefix:
    :param T final_values:
    :rtype: T
    """
    from Util import make_seq_of_type
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
    # TODO infer input_type from source layer class 'softmax' or 'linear' with 'log_softmax' activation
    if input_type == "prob":
      if source.output_before_activation:
        scores_in = source.output_before_activation.get_log_output()
      else:
        from TFUtil import safe_log
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
    :param TFNetwork network:
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

  def __init__(self, parent_rec_layer, **kwargs):
    self._parent_layers = {}  # type: typing.Dict[str,WrappedInternalLayer]
    with helper_variable_scope():
      self.parent_tile_multiples_var = tf.get_variable(
        name="parent_tile_multiples", shape=(), dtype=tf.int32, initializer=tf.ones_initializer())  # type: tf.Variable
    super(SubnetworkRecCellSingleStep, self).__init__(parent_rec_layer=parent_rec_layer, **kwargs)

  def set_parent_tile_multiple_op(self, multiple):
    """
    :param tf.Tensor multiplie:
    :rtype: tf.Operation
    """
    return tf.assign(self.parent_tile_multiples_var, multiple, name='set_parent_tile_multiple').op

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
    output = self.make_base_state_vars(layer_name, output)
    layer = WrappedInternalLayer(name=layer_name, network=self.parent_net, output=output, base_layer=layer)
    self._parent_layers[layer_name] = layer
    return layer

  def make_base_state_vars(self, name, output):
    """
    make state vars for input from outside the loop (automatic tile_batch)
    :param str name: layer name
    :param output: layer output data
    :rtype: converted output data 
    """
    rec_layer = self.parent_rec_layer
    from TFUtil import tile_transposed
    output.placeholder = tile_transposed(
      rec_layer.create_state_var(name="base_value_%s" % name, initial_value=output.placeholder, data_shape=output),
      axis=output.batch_dim_axis, 
      multiples=self.parent_tile_multiples_var.read_value())
    from TFUtil import DimensionTag
    for i, size in list(output.size_placeholder.items()):
      dim_tag = DimensionTag.get_tag_from_size_tensor(size)
      if not dim_tag:
        print("Warning, no defined dim tag on %r, axis %i" % (name, output.get_batch_axis(i)), file=log.v2)
        dim_tag = output.get_dim_tag(output.get_batch_axis(i))
        dim_tag.set_tag_on_size_tensor(size)
      new_size = rec_layer.create_state_var(name="base_size%i_%s" % (i, name), initial_value=size)
      new_size = tile_transposed(new_size, axis=0, multiples=self.parent_tile_multiples_var.read_value())
      dim_tag.set_tag_on_size_tensor(new_size)
      output.size_placeholder[i] = new_size
    return output

  def _get_extern_data(self, data, i):
    """
    :param dict[str,tf.Tensor] data: All data needed from outside of the loop. see _construct()
    :param tf.Tensor i: loop counter. scalar, int32, current step (time)
    """
    for key in data:
      if not key == 'source':
        self.net.extern_data.data[key].placeholder = data[key]
    # create base state vars for encodings
    if 'source' in data:
      rec_layer_input = self.parent_rec_layer.input_data.copy_as_batch_major()
      output = self.make_base_state_vars('source', rec_layer_input)
      # slice specific time step TODO put before tile batch
      self.net.extern_data.data['source'].placeholder = tf.gather(output.placeholder, i, axis=output.time_dim_axis)
     
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

    with tf.name_scope("state"):
      # See _SubnetworkRecCell.GetOutput.body.
      # net_vars is (prev_outputs_flat, prev_extra_flat), and prev_outputs_flat corresponds to self._initial_outputs,
      # where the keys are the layer names.
      # prev_extra is always expected to be batch-major.
      prev_outputs_data = []
      choice_dependent = []
      for k in sorted(self._initial_outputs.keys()):
        choice_dependent.append(False)
        for source_layer in self.net_dict[k]['from']:
          if source_layer in self.net_dict.keys() and self.net_dict[source_layer]['class'] == ChoiceStateVarLayer.layer_class:
            choice_dependent[-1] = True
        prev_outputs_data.append(self.layer_data_templates[k].output)

      def parse_recursive( nvs ):
        if isinstance(nvs, (tuple, list)):
          return [ parse_recursive(v) for v in nvs ]
        if isinstance(nvs, tf.Tensor):
          return False

      assert len(net_vars) == 2
      state_choice_dependent = parse_recursive( net_vars[1] )
      choice_dependent = (choice_dependent, state_choice_dependent)
      net_vars = rec_layer.create_state_vars_recursive(name_prefix="state", initial_values=net_vars, data_shape=(prev_outputs_data, None), choice_dependent=choice_dependent)

    # We are ignoring acc_tas (the tensor arrays).

    # Some layers make explicit use of the (global data) batch-dim,
    # which they can get via TFNetwork.get_data_batch_dim().
    # This will add a dependency on the external data, which we want to avoid.
    # We can avoid that by taking the batch dim instead from one of the other states.
    # Note that this would be wrong in beam search.
    self.net._batch_dim = rec_layer.get_batch_dim_from_state()

    # step info is needed to read corresponding encoding vectors 
    i = rec_layer.create_state_var("global_step", initial_value=i)

    with tf.name_scope("body"):
      res = body(i, net_vars, acc_tas, seq_len_info)
      assert len(res) == len(loop_vars)
      if len(res) == 3:
        i, net_vars, acc_tas = res
        seq_len_info = None
      else:
        i, net_vars, acc_tas, seq_len_info = res

    # make it setable by sprint
    rec_layer.set_state_var_final_value("global_step", i)

    rec_layer.set_state_vars_final_values_recursive("state", net_vars)
    return res


def main(argv):
  argparser = argparse.ArgumentParser(description='Compile some op')
  argparser.add_argument('config', help="filename to config-file")
  argparser.add_argument('--train', type=int, default=0, help='0 disable (default), 1 enable, -1 dynamic')
  argparser.add_argument('--eval', type=int, default=0, help='calculate losses. 0 disable (default), 1 enable')
  argparser.add_argument('--search', type=int, default=0, help='beam search. 0 disable (default), 1 enable')
  argparser.add_argument("--verbosity", default=4, type=int, help="5 for all seqs (default: 4)")
  argparser.add_argument("--summaries_tensor_name", help="create Tensor for tf.summary.merge_all()")
  argparser.add_argument("--rec_step_by_step", help="make step-by-step graph for this rec layer (eg. 'output')")
  argparser.add_argument("--rec_step_by_step_output_file", help="store meta info for rec_step_by_step (JSON)")
  argparser.add_argument("--output_file", help='output pb, pbtxt or meta, metatxt file')
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
    tf.set_random_seed(42)
    if args.train < 0:
      from TFUtil import get_global_train_flag_placeholder
      train_flag = get_global_train_flag_placeholder()
    else:
      train_flag = bool(args.train)
    eval_flag = bool(args.eval)
    search_flag = bool(args.search)
    network = create_graph(train_flag=train_flag, eval_flag=eval_flag, search_flag=search_flag, net_dict=net_dict)

    if args.rec_step_by_step:
      RecStepByStepLayer.post_compile(
        rec_layer_name=args.rec_step_by_step, network=network, output_file_name=args.rec_step_by_step_output_file)

    from TFNetworkLayer import LayerBase
    for layer in network.layers.values():
      assert isinstance(layer, LayerBase)
      if layer.output.time_dim_axis is None:
        continue
      with layer.cls_layer_scope(layer.name):
        tf.identity(layer.output.get_placeholder_as_batch_major(), name="output_batch_major")

    tf.group(*network.get_post_control_dependencies(), name="post_control_dependencies")

    if args.summaries_tensor_name:
      summaries_tensor = tf.summary.merge_all()
      assert isinstance(summaries_tensor, tf.Tensor), "no summaries in the graph?"
      tf.identity(summaries_tensor, name=args.summaries_tensor_name)

    if args.output_file and os.path.splitext(args.output_file)[1] in [".meta", ".metatxt"]:
      # https://www.tensorflow.org/api_guides/python/meta_graph
      saver = tf.train.Saver(
        var_list=network.get_saveable_params_list(), max_to_keep=2 ** 31 - 1)
      graph_def = saver.export_meta_graph()
    else:
      graph_def = graph.as_graph_def(add_shapes=True)

    print("Graph collection keys:", graph.get_all_collection_keys())
    print("Graph num operations:", len(graph.get_operations()))
    print("Graph def size:", Util.human_bytes_size(graph_def.ByteSize()))

    if args.output_file:
      filename = args.output_file
      _, ext = os.path.splitext(filename)
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
      from TFUtil import CollectionKeys
      with open(args.output_file_state_vars_list, "w") as f:
        for param in tf.get_collection(CollectionKeys.STATE_VARS):
          f.write("%s\n" % param.name)

 
if __name__ == '__main__':
  main(sys.argv)
