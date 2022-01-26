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
import contextlib
import tensorflow as tf
from tensorflow.python.framework import graph_io

import _setup_returnn_env  # noqa
import returnn.__main__ as rnn
from returnn.log import log
from returnn.config import Config
import returnn.util.basic as util
from returnn.util.basic import NotSpecified, Entity
import returnn.tf.compat as tf_compat
import returnn.tf.util.basic as tf_util
from returnn.tf.util.basic import Data, Dim, CollectionKeys
from returnn.tf.network import TFNetwork
from returnn.tf.layers.basic import LayerBase, register_layer_class
from returnn.tf.layers.base import WrappedInternalLayer
# noinspection PyProtectedMember
from returnn.tf.layers.rec import RecLayer, _SubnetworkRecCell


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


@contextlib.contextmanager
def helper_variable_scope():
  """
  :return: separate scope from the current name scope, such that variables are not treated as model params
  :rtype: tf.VariableScope
  """
  with tf_util.reuse_name_scope("IO", absolute=True) as scope:
    yield scope


class SubnetworkRecCellSingleStep(_SubnetworkRecCell):
  """
  Adapts :class:`_SubnetworkRecCell` such that we execute only a single step.
  """

  def __init__(self, **kwargs):
    self._parent_layers = {}  # type: typing.Dict[str,WrappedInternalLayer]
    self._parent_dim_tags = {}  # type: typing.Dict[Dim,Dim]
    self._parent_replace_deps = []  # type: typing.List[typing.Tuple[RecStepByStepLayer.StateVar,Data]]
    super(SubnetworkRecCellSingleStep, self).__init__(**kwargs)

  def _maybe_delay_tiled(self, state_var, output):
    """
    :param RecStepByStepLayer.StateVar state_var:
    :param Data output:
    :rtype: tf.Tensor
    """
    assert isinstance(state_var, RecStepByStepLayer.StateVar)
    assert isinstance(output, Data)
    rec_layer = self.parent_rec_layer
    assert isinstance(rec_layer, RecStepByStepLayer)
    if rec_layer.construction_state == rec_layer.ConstructionState.GetSources:
      self._parent_replace_deps.append((state_var, output))  # only in case the layer is accessed elsewhere later
      # No tiling yet because we cannot infer the in-loop batch dim. done in _construct.
      # Also use the state var read because we will use this later inside the loop.
      x = state_var.read()
    elif rec_layer.construction_state == rec_layer.ConstructionState.Init:
      self._parent_replace_deps.append((state_var, output))
      with tf.control_dependencies([state_var.init_op()]):
        x = tf.identity(
          output.placeholder,
          name="state_var_before_loop_depend_on_init__%s" % tf_util.get_valid_scope_name_from_str(state_var.name))
    elif rec_layer.construction_state == rec_layer.ConstructionState.InLoop:
      x = self._tiled(output, state_var.read())
    else:
      raise ValueError("unexpected construction state %r" % rec_layer.construction_state)
    dim_tag = Dim.get_tag_from_size_tensor(output.placeholder)
    if dim_tag:
      dim_tag.set_tag_on_size_tensor(x, same_as_before=True)
    output.placeholder = x
    return x

  def _tiled(self, output, x):
    """
    :param Data output:
    :param tf.Tensor x:
    :rtype: tf.Tensor
    """
    rec_layer = self.parent_rec_layer
    assert isinstance(rec_layer, RecStepByStepLayer)
    return tf_util.tile_transposed(
      x, axis=output.batch_dim_axis,
      multiples=rec_layer.get_parent_tile_multiples())

  def get_parent_dim_tag(self, dim_tag):
    """
    :param Dim dim_tag:
    :rtype: Dim
    """
    if dim_tag.dimension is not None:
      return dim_tag  # as-is, not dynamic
    if dim_tag.is_batch_dim():
      return dim_tag  # as-is
    if dim_tag in self._parent_dim_tags:
      return self._parent_dim_tags[dim_tag]
    rec_layer = self.parent_rec_layer
    assert isinstance(rec_layer, RecStepByStepLayer)
    new_size, state_var = rec_layer.create_state_var(
      name="base_size_%s" % tf_util.get_valid_scope_name_from_str(dim_tag.description),
      initial_value=dim_tag.dyn_size)
    dim_tag_dyn_size_ext = dim_tag.dyn_size_ext.copy()
    with tf_util.same_control_flow_ctx(dim_tag_dyn_size_ext.placeholder), tf_util.reuse_name_scope("", absolute=True):
      # Need to create new tensor currently for the set_tag_on_size_tensor logic.
      # (set_tag_on_size_tensor is going to be removed at some point but currently we need it.)
      dim_tag_dyn_size_ext.placeholder = tf.identity(
        dim_tag_dyn_size_ext.placeholder, name=dim_tag_dyn_size_ext.placeholder.op.name + "_copy_new_dim_tag")
    # Copy dim tag.
    dim_tag_ = Dim(
      kind=dim_tag.kind, description=dim_tag.description + "_rec_step_by_step",
      dimension=None, dyn_size_ext=dim_tag_dyn_size_ext,
      batch=dim_tag.batch)
    dim_tag_.set_tag_on_size_tensor(dim_tag_dyn_size_ext.placeholder)
    self._maybe_delay_tiled(state_var, dim_tag_.dyn_size_ext)
    self._parent_dim_tags[dim_tag] = dim_tag_
    return dim_tag_

  def _get_parent_layer(self, layer_name):
    """
    :param str layer_name: without "base:" prefix
    :rtype: WrappedInternalLayer
    """
    if layer_name in self._parent_layers:
      return self._parent_layers[layer_name]
    layer = super(SubnetworkRecCellSingleStep, self)._get_parent_layer(layer_name)
    rec_layer = self.parent_rec_layer
    if rec_layer is None:  # at template construction
      return layer
    assert isinstance(rec_layer, RecStepByStepLayer)
    output = layer.output.copy()
    new_dim_tags = []
    for i, dim_tag in enumerate(output.dim_tags):
      new_dim_tags.append(self.get_parent_dim_tag(dim_tag))
    output = output.copy_template_new_dim_tags(new_dim_tags=new_dim_tags, keep_special_axes=True)
    output.placeholder = layer.output.placeholder
    x, state_var = rec_layer.create_state_var(
      name="base_value_%s" % layer_name, initial_value=output.placeholder, data_shape=output)
    self._maybe_delay_tiled(state_var, output)
    layer = WrappedInternalLayer(name=layer_name, network=self.parent_net, output=output, base_layer=layer)
    self._parent_layers[layer_name] = layer
    return layer

  def get_sources(self, sources):
    """
    :param list[str] sources:
    :rtype: list[WrappedInternalLayer]
    """
    return [self._get_parent_layer(layer_name) for layer_name in sources]

  def _set_construction_state_in_loop(self):
    rec_layer = self.parent_rec_layer
    assert isinstance(rec_layer, RecStepByStepLayer)
    rec_layer.set_construction_state_in_loop()

    for args in self._parent_replace_deps:
      self._maybe_delay_tiled(*args)

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
    i, _ = rec_layer.create_state_var("i", i)
    with tf.name_scope("state"):
      # See _SubnetworkRecCell.GetOutput.body.
      # net_vars is (prev_outputs_flat, prev_extra_flat), and prev_outputs_flat corresponds to self._initial_outputs,
      # where the keys are the layer names.
      # prev_extra is always expected to be batch-major.
      prev_outputs_data = [self.layer_data_templates[k].output for k in sorted(self._initial_outputs.keys())]
      net_vars = rec_layer.create_state_vars_recursive(
        name_prefix="state", initial_values=net_vars, data_shape=(prev_outputs_data, None))
    # We are ignoring acc_tas (the tensor arrays).

    self._set_construction_state_in_loop()

    with tf.name_scope("cond"):
      rec_layer.create_state_var("cond", tf.constant(True))
      rec_layer.set_state_var_final_value("cond", cond(i, net_vars, acc_tas, seq_len_info, allow_inf_max_len=True))

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

  def _construct(self, prev_outputs, prev_extra, i, data=None,
                 inputs_moved_out_tas=None, needed_outputs=("output",)):
    """
    This is called from within the `tf.while_loop` of the :class:`RecLayer`,
    to construct the subnetwork, which is performed step by step.

    :param dict[str,tf.Tensor] prev_outputs:
    :param dict[str,dict[str,tf.Tensor]] prev_extra:
    :param tf.Tensor i: loop counter. scalar, int32, current step (time)
    :param dict[str,tf.Tensor] data: via tensor arrays
    :param dict[str,tf.TensorArray]|None inputs_moved_out_tas:
    :param set[str] needed_outputs:
    """
    # The TensorArrays depend on the base network (base state vars)
    # which would not have tiling enabled at the time they were constructed.
    # So we add tiling here.
    assert data is not None
    for key, value in list(data.items()):
      if key != "source":
        continue  # ignore other sources currently...
      assert key in self.net.extern_data.data
      data_ = self.net.extern_data.data[key]
      data[key] = self._tiled(data_, value)

    super(SubnetworkRecCellSingleStep, self)._construct(
      prev_outputs=prev_outputs, prev_extra=prev_extra, i=i, data=data,
      inputs_moved_out_tas=inputs_moved_out_tas, needed_outputs=needed_outputs)


class RecStepByStepLayer(RecLayer):
  """
  Represents a single step of :class:`RecLayer`.
  The purpose is to execute a single step only.
  This also takes care of all needed state, and stochastic (maybe latent) variables (via :class:`ChoiceLayer`).
  All the state is kept in *state variables*, such that you can avoid feeding/fetching.
  Instead, any decoder implementation using this must explicitly assign the state variables.
  Stochastic variables (:class:`ChoiceLayer`) are breakpoints where an external application
  can implement custom logic to select hypotheses.
  So this can be used to implement beam search, or to implement a custom decoder in an external application.
  RASR (Sprint) is one such example which makes use of this for decoding.

  The necessary meta information about the list of state vars,
  ops for specific calculation steps, etc.
  is stored in TF graph collections and also in a JSON file.
  RASR currently only uses the TF graph collections.

  There are different kinds of state vars:
  - Base state vars (encoder or so), depends only on the input data, not updated by the decoder loop.
  - decoder_input_vars: Stochastic choice state vars, updated by the decoder loop.
  - decoder_output_vars: Stochastic scores state vars, updated by the decoder loop.
  - (Deterministic) loop state vars, updated by the decoder loop.

  RASR does the following logic::

    # initial state vars
    encode_ops(input_placeholders=...)  # -> assign (base and loop) state_vars

    # Main decoder loop
    while seq_not_ended(...):

      for hyp in current_hyps:  # (in practice, this is partially batched)

        state_vars.assign(...)  # for current hyp
        decoder_input_vars.assign(...)  # for current hyp, the previous choice

        update_ops(decoder_input_vars)  # -> assign/update potentially some loop state vars

        # Loop over stochastic vars, and call decode_ops for each.
        for (choices, scores, decode_op) in zip(decoder_input_vars, decoder_output_vars, decode_ops):

          for all_possible_succeeding_input_vars:  # loop not needed for first stochastic var

            decoder_input_vars[:i].assign(...)  # ...

            # decoder_input_vars contains prev choices
            decode_ops(state_vars, decoder_input_vars)  # -> assign scores

        decoder_input_vars.assign(...)  # for current hyp, (still!) the previous (!) choice

        post_update_ops(state_vars, decoder_input_vars)  # -> assign new state_vars

      < select new hyps and prune some away, based on scores >

  Because RASR is the main application, we adopt the recurrence logic of the RecLayer to be compatible to RASR.

  E.g., if you want to implement beam search in an external application which uses the compiled graph,
  you would compile the graph with search_flag disabled (such that RETURNN does not do any related logic),
  enable this recurrent step-by-step compilation, and then do the following TF session runs:

  * Call the initializers of all the state variables, which also includes everything from the base (e.g. encoder),
    while feeding in the placeholders for the input features.
    This is "init_op" in the info json, "encode_ops" in the TF graph collections.
    All further session runs should not need any feed values. All needed state should be in state vars.
    This will init all state vars, except stochastic_var_*.
  * For each decoder step:
    * For each stochastic variable (both latent variables and observable variables),
      in the order as in "stochastic_var_order" in the info json:
      - This calculation depends on all state vars
        (except of stochastic_var_scores_* and only the dependent other stochastic_var_choice_*).
        I.e. all those state vars must be assigned accordingly.
      - Calculate `"stochastic_var_scores_%s" % name` (which are the probabilities in +log space).
        This is "calc_scores_op" in the info json, "decode_ops" in the TF graph collections.
      - Do a choice, build up a beam of hypotheses.
      - Set the stochastic state var `"stochastic_var_choice_%s" % name` to the selected values (label indices).
      - If the beam has multiple items, i.e. the batch dimension changed, you must make sure
        that all further used state variables will also have the same batch dim.
    * Do a single session run for the next values of these state vars:
      "i", "end_flag" (if existing), "dyn_seq_len" (if existing), "state_*" (multiple vars).
      These are also the state vars which will get updated in every further recurrent step.
      The "base_*" state vars are always kept (although you might need to update the batch dim),
      and the "stochastic_var_*" state vars have the special logic for the stochastic variables.
      This is "next_step_op" in the info json.

  Thus, the info json should contain the following:

  * "state_vars": dict[str, dict[str, Any]]. name -> dict with:
    * "var_op": str
    * "shape": tuple[int|None,...]. including batch dim. always batch-major, or scalar.
    * "dtype": str
    The state vars usually are:
    * "i"
    * "end_flag" (if existing)
    * "dyn_seq_len" (if existing)
    * "state_*" (multiple vars)
    * "base_*" (multiple vars). only once via init_op, maybe then batch-tiled, but otherwise not changed for once seq.
    * "stochastic_var_*" (multiple vars)
    However, no specific assumptions should be made on any of these.
    It should not matter to the decoder what we have here.
  * "stochastic_vars": dict[str,dict[str, str]]. name -> dict with:
    * "calc_scores_op": str. op to calculate the scores state var
    * "scores_state_var": str. == "stochastic_var_scores_%s" % name
    * "choice_state_var": str. == "stochastic_var_choice_%s" % name
  * "stochastic_var_order": list[str]. the order of the stochastic vars.

  * "init_op": str. op. the initializer for all state vars, including encoder.
  * "next_step_op: str. op. the update op for all state vars.

  """

  layer_class = "rec_step_by_step"
  SubnetworkRecCell = SubnetworkRecCellSingleStep

  class ConstructionState:
    """construction states"""
    GetSources = Entity("get_sources")
    Init = Entity("init")
    InLoop = Entity("in_body")

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
    for key, layer_dict in list(rec_layer_dict["unit"].items()):
      assert isinstance(layer_dict, dict)
      if layer_dict["class"] == "choice":
        layer_dict["class"] = ChoiceStateVarLayer.layer_class
    rec_layer_dict["class"] = RecStepByStepLayer.layer_class

  @classmethod
  def post_compile(cls, rec_layer_name, network, output_file_name=None):
    """
    :param str rec_layer_name:
    :param TFNetwork network:
    :param str|None output_file_name: via command line --rec_step_by_step_output_file
    """
    assert rec_layer_name in network.layers
    rec_layer = network.layers[rec_layer_name]
    assert isinstance(rec_layer, RecStepByStepLayer)
    cell = rec_layer.cell
    assert isinstance(cell, SubnetworkRecCellSingleStep)
    rec_sub_net = cell.net
    assert isinstance(rec_sub_net, TFNetwork)
    info = {"state_vars": {}, "stochastic_var_order": [], "stochastic_vars": {}}

    print("State vars:")
    for name, var in sorted(rec_layer.state_vars.items()):
      assert isinstance(name, str)
      assert isinstance(var, RecStepByStepLayer.StateVar)
      print(" %s: %r, shape %s, dtype %s" % (name, var.var.op.name, var.var.shape, var.var.dtype.base_dtype.name))
      info["state_vars"][name] = {
        "var_op": var.var.op.name,
        "shape": [int(d) if d is not None else None for d in var.var_data_shape.batch_shape],
        "dtype": var.var.dtype.base_dtype.name}

    # global_vars were used by the computation graph in some earlier version.
    # We do not make use of global_vars anymore because this is redundant with the state vars.
    # We anyway create the collection here such that older RASR binaries still work fine.
    # See https://github.com/rwth-i6/returnn/pull/874.
    tf_compat.v1.get_collection_ref("global_vars")

    # Encoder and all decoder state initializers.
    encode_ops_coll = tf_compat.v1.get_collection_ref("encode_ops")
    # We must make sure that the non-base loop vars and all their dependencies,
    # are initialized after we have initialized the base loop vars.
    # A control dependency on the var.init_op is not enough as the init_op is only the tf.assign, not more.
    # We make sure in the get_parent_layer via the construction_state.
    init_ops = []
    for name, var in sorted(rec_layer.state_vars.items()):
      assert isinstance(name, str)
      assert isinstance(var, RecStepByStepLayer.StateVar)
      if not name.startswith("stochastic_var_"):
        init_ops.append(var.init_op())
    init_op = tf.group(*init_ops, name="rec_step_by_step_init_op")
    info["init_op"] = init_op.name
    encode_ops_coll.append(init_op)

    # Based on some decoder state and encoder, calculate the next scores for the next stochastic variable.
    decode_ops_coll = tf_compat.v1.get_collection_ref("decode_ops")  # calculates the scores
    decoder_output_vars_coll = tf_compat.v1.get_collection_ref("decoder_output_vars")  # scores
    decoder_input_vars_coll = tf_compat.v1.get_collection_ref("decoder_input_vars")  # choices
    print("Stochastic vars, and their order:")
    # additional flexibility to match RASR's specific order of decoder inputs feeding
    if rec_layer.reverse_stochastic_var_order:
      stochastic_var_order = rec_layer.stochastic_var_order[::-1]
    else:
      stochastic_var_order = rec_layer.stochastic_var_order
    for name in stochastic_var_order:
      print(" %s" % name)
      info["stochastic_var_order"].append(name)
      choice_layer = rec_sub_net.layers[name]
      assert isinstance(choice_layer, ChoiceStateVarLayer)
      # Usually all choices depend on the scores.
      # However, some legacy configs used prev choices which are fed as-is because they are known.
      # See comment in ChoiceStateVarLayer.
      if choice_layer.score_dependent:
        calc_scores_op = rec_layer.state_vars["stochastic_var_scores_%s" % name].final_op()
        info["stochastic_vars"][name] = {
          "calc_scores_op": calc_scores_op.name,
          "scores_state_var": "stochastic_var_scores_%s" % name,
          "choice_state_var": "stochastic_var_choice_%s" % name}
        decode_ops_coll.append(calc_scores_op)
        decoder_output_vars_coll.append(rec_layer.state_vars["stochastic_var_scores_%s" % name].var)
      else:
        info["stochastic_vars"][name] = {"choice_state_var": "stochastic_var_choice_%s" % name}
      decoder_input_vars_coll.append(rec_layer.state_vars["stochastic_var_choice_%s" % name].var)

    # We do not make use of update_ops anymore. This was used as a separate step
    # such that only "choice-dependent" state vars were updated separately in the step before.
    # However, this logic was incomplete and broken in general,
    # and also the optimization only lead to negligible speedup, so we removed it.
    # We anyway create the collection here such that older RASR binaries still work fine.
    # See https://github.com/rwth-i6/returnn/pull/874 for some discussion.
    tf_compat.v1.get_collection_ref("update_ops")
    # Based on the decoder state, encoder, and choices, calculate the next state.
    post_update_ops_coll = tf_compat.v1.get_collection_ref("post_update_ops")
    state_vars_coll = tf_compat.v1.get_collection_ref(CollectionKeys.STATE_VARS)
    next_step_ops = []
    for name, var in sorted(rec_layer.state_vars.items()):
      assert isinstance(name, str)
      assert isinstance(var, RecStepByStepLayer.StateVar)
      if not name.startswith("stochastic_var_") and not name.startswith("base_"):
        next_step_ops.append(var.final_op())
        state_vars_coll.append(var.var)
    next_step_op = tf.group(*next_step_ops, name="rec_step_by_step_post_update_op")
    info["next_step_op"] = next_step_op.name
    post_update_ops_coll.append(next_step_op)

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
      dyn_one = tf_compat.v1.placeholder_with_default(tf.constant(1), ())
      zero_initializer = tf.zeros(
        [d if (d is not None) else dyn_one for d in self.var_data_shape.batch_shape],
        dtype=self.var_data_shape.dtype)
      zero_initializer.set_shape(self.var_data_shape.batch_shape)
      with helper_variable_scope():
        self.var = tf_compat.v1.get_variable(
          name=name, initializer=zero_initializer, validate_shape=False)  # type: tf.Variable
      self.var.set_shape(self.var_data_shape.batch_shape)
      self._init_op = None
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
      if self._init_op:
        return self._init_op
      assert self.var_initial_value is not None
      self._init_op = tf_compat.v1.assign(self.var, self.var_initial_value, name="init_state_var_%s" % self.name).op
      return self._init_op

    def final_op(self):
      """
      :return: op which does self.var.assign(self.final_value) (final value maybe converted)
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
      assert not path, (
        "There should be no path from extern data to %s final op value, but there is:\n%s" % (
          self, "\n".join(map(repr, path))))
      if self.orig_data_shape.batch_dim_axis not in (0, None):
        x = self.orig_data_shape.copy()
        x.placeholder = value
        x = x.copy_compatible_to(self.var_data_shape)
        value = x.placeholder
      return tf_compat.v1.assign(self.var, value, name="final_state_var_%s" % self.name).op

  def __init__(self, _orig_sources, sources, network, name, output, unit, axis,
               reverse_stochastic_var_order=False, **kwargs):
    """
    :param str|list[str]|None _orig_sources:
    :param list[LayerBase] sources:
    :param returnn.tf.network.TFNetwork network:
    :param str name:
    :param Data output:
    :param SubnetworkRecCellSingleStep unit:
    :param Dim axis:
    :param bool reverse_stochastic_var_order:
    """
    assert isinstance(unit, SubnetworkRecCellSingleStep)
    kwargs = kwargs.copy()
    kwargs["optimize_move_layers_out"] = False
    self.state_vars = {}  # type: typing.Dict[str,RecStepByStepLayer.StateVar]
    self.stochastic_var_order = []  # type: typing.List[str]
    self._parent_tile_multiples = None  # type: typing.Optional[tf.Tensor]
    self.construction_state = self.ConstructionState.GetSources
    self.reverse_stochastic_var_order = reverse_stochastic_var_order

    # Some early assigns needed for set_parent_layer, and that __repr__ works.
    self.network = network
    self.name = name
    self.output = output
    unit.set_parent_layer(self)  # early assign, so that unit.get_parent_layer() works
    if _orig_sources:
      # Make sure we resolve sources again through our get_parent_layer logic.
      if isinstance(_orig_sources, (list, tuple)):
        assert len(sources) == len(_orig_sources)
      else:
        assert isinstance(_orig_sources, str)
        assert len(sources) == 1
        _orig_sources = [_orig_sources]
      assert unit.parent_rec_layer is self  # this is needed such that get_sources and get_parent_layer works
      # get_sources will call get_parent_layer which will register it properly as a base state var.
      # Note that we are not yet inside the loop, thus this will not enable tiling,
      # and thus the resulting TensorArray will not have tiling.
      # Thus, we will post-edit the subnet extern_data ("data:source") in our derived _construct to add tiling.
      sources = unit.get_sources(_orig_sources)
    else:
      assert not sources

    # axis might be an existing axis from the encoder (e.g. transducer),
    # so we also need to wrap it.
    assert isinstance(axis, Dim)
    if axis and sources and sources[0].base_layer.output.have_dim_tag(axis):
      base_axis = axis
      axis = unit.get_parent_dim_tag(base_axis)
      unit.time_dim_tag = axis
      # noinspection PyProtectedMember
      unit._time_dim_tags.add(axis)
      unit.net._inside_rec_time_dim = axis
      if output.have_dim_tag(base_axis):
        output = output.copy_template_replace_dim_tag(
          axis=output.get_axis_from_description(base_axis),
          new_dim_tag=axis)

    self.construction_state = self.ConstructionState.Init
    if self.have_base_state_vars():
      self._set_global_batch_dim(self.get_batch_dim_from_base_state_var())
    super(RecStepByStepLayer, self).__init__(
      sources=sources, network=network, name=name, output=output, unit=unit, axis=axis, **kwargs)

  def set_construction_state_in_loop(self):
    """
    Set that we entered the body.
    """
    self.construction_state = self.ConstructionState.InLoop

    # Some layers make explicit use of the (global data) batch-dim,
    # which they can get via TFNetwork.get_data_batch_dim().
    # This will add a dependency on the external data, which we want to avoid.
    # We can avoid that by taking the batch dim instead from one of the other states.
    # Note that this would be wrong in beam search.
    self._set_global_batch_dim(self.get_batch_dim_from_loop_state_var())

  def _set_global_batch_dim(self, batch_dim):
    """
    :param tf.Tensor batch_dim:
    """
    self.network.get_global_batch_info().dim = batch_dim

  def get_parent_tile_multiples(self):
    """
    :rtype: tf.Tensor
    """
    if self._parent_tile_multiples is not None:
      return self._parent_tile_multiples
    with tf_util.reuse_name_scope("parent_tile_multiples", absolute=True):
      base_batch_dim = self.get_batch_dim_from_base_state_var()
      loop_batch_dim = self.get_batch_dim_from_loop_state_var()
      with tf.control_dependencies([
            tf.Assert(tf.equal(loop_batch_dim % base_batch_dim, 0),
                      ["loop_batch_dim", loop_batch_dim, "base_batch_dim", base_batch_dim])]):
        self._parent_tile_multiples = loop_batch_dim // base_batch_dim
    return self._parent_tile_multiples

  def create_state_var(self, name, initial_value=None, data_shape=None):
    """
    A state var is a variable where the initial value is given by the encoder, or a constant,
    and the final value is determined by one step of this rec layer (usually called the decoder).

    :param str name:
    :param tf.Tensor|None initial_value: assumes batch-major, if data_shape is not given
    :param Data|None data_shape:
    :rtype: (tf.Tensor,RecStepByStepLayer.StateVar)
    """
    assert name not in self.state_vars
    assert data_shape or initial_value is not None
    if data_shape:
      assert isinstance(data_shape, Data)
      if data_shape.have_batch_axis() and initial_value is not None:
        assert initial_value.shape.dims[data_shape.batch_dim_axis].value is None
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
    return var.read(), var

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
      x, _ = self.create_state_var(name=name_prefix, initial_value=initial_values, data_shape=data_shape)
      return x
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

  def have_base_state_vars(self):
    """
    :rtype: bool
    """
    for name, _ in sorted(self.state_vars.items()):
      assert isinstance(name, str)
      if name.startswith("base_"):
        return True
    return False

  def get_batch_dim_from_base_state_var(self):
    """
    :return: batch-dim, from some (any) base state var, scalar, int32
    :rtype: tf.Tensor|int
    """
    for name, v in sorted(self.state_vars.items()):
      assert isinstance(name, str)
      assert isinstance(v, RecStepByStepLayer.StateVar)
      if name.startswith("base_"):
        if v.var_data_shape.have_batch_axis():
          with tf_util.reuse_name_scope("batch_dim_from_base_state_var_%s" % v.name, absolute=True):
            return tf.shape(v.var.value())[v.var_data_shape.batch_dim_axis]
    raise Exception("None of the base state vars do have a batch-dim: %s" % self.state_vars)

  def get_batch_dim_from_loop_state_var(self):
    """
    :return: batch-dim, from some (any) loop state var, scalar, int32
    :rtype: tf.Tensor|int
    """
    for name, v in sorted(self.state_vars.items()):
      assert isinstance(v, RecStepByStepLayer.StateVar)
      if not name.startswith("base_") and not name.startswith("stochastic_var_"):
        if v.var_data_shape.batch_dim_axis is not None:
          with tf_util.reuse_name_scope("batch_dim_from_state_%s" % v.name, absolute=True):
            return tf.shape(v.var.value())[v.var_data_shape.batch_dim_axis]
    raise Exception("None of the loop state vars do have a batch-dim: %s" % self.state_vars)

  def add_stochastic_var(self, name):
    """
    :param str name:
    """
    assert name not in self.stochastic_var_order
    self.stochastic_var_order.append(name)

  @classmethod
  def transform_config_dict(cls, d, network, get_layer):
    """
    :param dict[str] d: will modify inplace
    :param returnn.tf.network.TFNetwork network:
    :param ((str) -> LayerBase) get_layer: function to get or construct another layer
    """
    d["_orig_sources"] = d.get("from", "data")  # used for potential custom get_parent_layer of SubnetworkRecCell
    super(RecStepByStepLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)


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
               explicit_search_sources=None, score_dependent=True,
               **kwargs):
    super(ChoiceStateVarLayer, self).__init__(**kwargs)
    rec_layer = self.network.parent_layer
    assert isinstance(rec_layer, RecStepByStepLayer)
    # for compatibility: one way to set the ngram label context via additional choices
    # but older history is not score-dependent stochastic_var anymore
    self.score_dependent = score_dependent
    if score_dependent:
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
    self.output.placeholder, _ = rec_layer.create_state_var(
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

  @classmethod
  def get_out_data_from_opts(cls, name, sources, target, network, **kwargs):
    """
    :param str name:
    :param list[LayerBase] sources:
    :param str target:
    :param returnn.tf.network.TFNetwork network:
    :rtype: Data
    """
    # This is simplified from ChoiceLayer.get_out_data_from_opts.
    target = target[0] if isinstance(target, list) else target  # only the first matters here
    if target:
      out_data = cls._static_get_target_value(
        target=target, network=network, mark_data_key_as_used=False).copy_template(name="%s_output" % name)
      out_data.available_for_inference = True  # in inference, we would do search
    else:  # no target. i.e. we must do search
      # Output will be the sparse version of the input.
      out_data = sources[0].output.copy_template().copy_as_batch_major()
      dim_tags = list(out_data.dim_tags)
      del dim_tags[out_data.feature_dim_axis]
      out_data = Data(
        name="%s_output" % name, dim_tags=dim_tags, sparse=True, dim=out_data.dim,
        batch=out_data.batch.copy_set_beam(None) if out_data.batch else network.get_global_batch_info())
    return out_data


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
      with tf_util.reuse_name_scope(layer.get_absolute_name_scope_prefix()[:-1], absolute=True):
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
