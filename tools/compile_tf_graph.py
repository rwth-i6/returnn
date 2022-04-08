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
from tensorflow.python.util import nest

import _setup_returnn_env  # noqa
import returnn.__main__ as rnn
from returnn.log import log
from returnn.config import Config
import returnn.util.basic as util
from returnn.util.basic import NotSpecified, Entity
import returnn.tf.compat as tf_compat
import returnn.tf.util.basic as tf_util
from returnn.tf.util.basic import Data, Dim, CollectionKeys
from returnn.tf.network import TFNetwork, ExternData
from returnn.tf.layers.basic import LayerBase, register_layer_class, SourceLayer
from returnn.tf.layers.base import WrappedInternalLayer
# noinspection PyProtectedMember
from returnn.tf.layers.rec import RecLayer, _SubnetworkRecCell, _TemplateLayer


config = None  # type: typing.Optional[Config]


def init(config_filename, log_verbosity, device):
  """
  :param str config_filename: filename to config-file
  :param int log_verbosity:
  :param str device:
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
    "device": device,
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
  Used by :class:`RecStepByStepLayer`. See :class:`RecStepByStepLayer` for further documentation.
  """

  def __init__(self, **kwargs):
    self._parent_layers = {}  # type: typing.Dict[str,WrappedInternalLayer]
    self._parent_dim_tags = {}  # type: typing.Dict[Dim,Dim]
    self._parent_replace_deps = []  # type: typing.List[typing.Tuple[RecStepByStepLayer.StateVar,Data]]
    super(SubnetworkRecCellSingleStep, self).__init__(**kwargs)
    extern_data_copy = ExternData()
    extern_data_copy.data.update({k: v.copy_template() for (k, v) in self.net.extern_data.data.items()})
    self.net_delayed_update = TFNetwork(
      name="%s(delayed-update)" % self.net.name,
      extern_data=extern_data_copy,
      train_flag=self.net.train_flag,
      search_flag=self.net.search_flag,
      eval_flag=False,
      inside_rec_time_dim=self.time_dim_tag,
      control_flow_ctx=self.net.control_flow_ctx,
      absolute_name_prefix=self.net.get_absolute_name_prefix(),
      parent_net=self.parent_net)
    self.net_delayed_update.is_root_in_ctx = True
    self.delayed_state_update_op = None  # type: typing.Optional[tf.Operation]
    self.state_update_op = None  # type: typing.Optional[tf.Operation]

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
    state_var = rec_layer.create_state_var(
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
      kind=dim_tag.kind, description=dim_tag.description + "_base_state_var",
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
    state_var = rec_layer.create_state_var(
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
    cell = self
    rec_layer = self.parent_rec_layer
    assert isinstance(rec_layer, RecStepByStepLayer)
    self.net_delayed_update.parent_layer = rec_layer

    if len(loop_vars) == 3:
      i, net_vars, acc_tas = loop_vars
      seq_len_info = None
    else:
      i, net_vars, acc_tas, seq_len_info = loop_vars
      seq_len_info_ = rec_layer.create_state_vars_recursive(("end_flag", "dyn_seq_len"), seq_len_info)
      seq_len_info = nest.map_structure(lambda state_var: state_var.read(), seq_len_info_)
    initial_i = i
    i = rec_layer.create_state_var("i", initial_i).read()

    # Go through layers with state, check their (direct + indirect) dependencies,
    # whether they depend on any choice vars.
    layers_with_state = set(self._initial_outputs.keys()).union(self._initial_extra_outputs.keys())
    layers_cur_iteration = set()
    layers_delayed = set()
    layer_deps_by_layer = {}  # type: typing.Dict[str, typing.Set[str]]
    layers_delayed_prev_deps = set()
    choice_layers = set()
    for layer_name in layers_with_state:
      template_layer = self.layer_data_templates[layer_name]
      queue = [template_layer]
      visited = set()
      prev_frame_deps = set()
      choice_deps = set()
      source_deps = set()
      while queue:
        cur = queue.pop(0)
        if cur in visited:
          continue
        visited.add(cur)
        if cur.layer_class_type is ChoiceStateVarLayer:
          choice_deps.add(cur.name)
          continue
        if cur.layer_class_type is SourceLayer:
          source_deps.add(cur.name)
          continue
        for dep in cur.cur_frame_dependencies:
          if dep in visited:
            continue
          if self.net not in dep.network.get_network_hierarchy():
            continue  # ignore layers from parent network
          assert isinstance(dep, _TemplateLayer)
          queue.append(dep)
        prev_frame_deps.update([dep.name for dep in cur.prev_frame_dependencies])
      layer_deps_by_layer[layer_name] = prev_frame_deps
      if not choice_deps:
        layers_cur_iteration.add(layer_name)
      else:
        print(
          "Delayed: Layer %r depends on choices %r, deps on prev frame %r" % (
            layer_name, choice_deps, prev_frame_deps), file=log.v4)
        if source_deps:
          raise NotImplementedError("Delayed layers with source dependencies (%s) not supported yet" % source_deps)
        layers_delayed.add(layer_name)
        layers_delayed_prev_deps.update(prev_frame_deps)
        choice_layers.update(choice_deps)
    for choice_layer in choice_layers:
      assert choice_layer not in layers_cur_iteration
      layers_delayed.add(choice_layer)  # if not yet added, add now because we anyway need them
      rec_layer.add_stochastic_var(choice_layer)

    # See _SubnetworkRecCell.get_output().body().
    # net_vars is (prev_outputs_flat, prev_extra_flat).
    # prev_outputs_flat corresponds to the dict self._initial_outputs, specifically sorted(self._initial_outputs)
    # where the keys are the layer names.
    # prev_extra is always expected to be batch-major.
    # prev_extra_flat corresponds to the dict self._initial_extra_outputs,
    # specifically sorted(self._initial_extra_outputs),
    # where the keys are the layer names.
    init_outputs_flat, init_extra_flat = net_vars
    assert len(init_outputs_flat) == len(self._initial_outputs)
    assert len(init_extra_flat) == len(self._initial_extra_outputs)
    init_outputs = {k: v for k, v in zip(sorted(self._initial_outputs.keys()), init_outputs_flat)}
    init_extra = {
      k: util.dict_zip(sorted(self._initial_extra_outputs[k]), v)
      for (k, v) in zip(sorted(self._initial_extra_outputs), init_extra_flat)}

    # noinspection PyShadowingNames
    class _LayerStateHelper:
      def __init__(self, layer_name, prefix):
        """
        :param str layer_name:
        :param str prefix:
        """
        self.layer_name = layer_name
        self.prefix = prefix

        initial_values = {}
        data_shapes = {}
        if layer_name in init_outputs:
          initial_values["output"] = init_outputs[layer_name]
          data_shapes["output"] = cell.layer_data_templates[layer_name].output
        if layer_name in init_extra:
          initial_values["extra"] = init_extra[layer_name]
          data_shapes["extra"] = nest.map_structure(lambda s: None, init_extra[layer_name])  # should be batch-major
        assert isinstance(rec_layer, RecStepByStepLayer)
        self.state_vars = rec_layer.create_state_vars_recursive(
          name_prefix="%s/%s" % (prefix, layer_name),
          initial_values=initial_values, data_shapes=data_shapes)
        self._reads_once = None

      def reset_reads_once(self):
        """
        Next get_reads_once will create new op.
        """
        self._reads_once = None

      def get_reads_once(self):
        """
        :return: same structure as state vars with the actual reads, type tf.Tensor. makes sure they are created once
        """
        if self._reads_once is not None:
          return self._reads_once
        self._reads_once = self.reads()
        return self._reads_once

      def reads(self):
        """
        :return: same structure as state vars with the actual reads, type tf.Tensor.
        """
        return nest.map_structure(lambda state_var: state_var.read(), self.state_vars)

      def assigns_flat(self, state):
        """
        :param state: same structure as state vars
        :return: list of tf.Assign ops
        :rtype: list[tf.Operation]
        """
        def _map(state_var, state_value):
          assert isinstance(state_var, RecStepByStepLayer.StateVar)
          assert isinstance(state_value, tf.Tensor)
          return state_var.assign(state_value)

        assert isinstance(rec_layer, RecStepByStepLayer)
        nest.assert_same_structure(state, self.state_vars)
        return nest.flatten(nest.map_structure(_map, self.state_vars, state))

      @staticmethod
      def get_from_net(net, layer_name, prev=False):
        """
        :param TFNetwork net:
        :param str layer_name:
        :param bool prev:
        :return: same structure as state vars, type tf.Tensor
        """
        layer = net.layers[("prev:" + layer_name) if prev else layer_name]
        out = {}
        if layer_name in init_outputs:
          out["output"] = layer.output.placeholder
        if layer_name in init_extra:
          out["extra"] = layer.rec_vars_outputs
        return out

    with tf.name_scope("state_delayed"):
      layers_prev_prev = {}
      for layer_name in layers_delayed.union(layers_delayed_prev_deps):
        layers_prev_prev[layer_name] = _LayerStateHelper(layer_name, "state_delayed")

    with tf.name_scope("state"):
      layers_prev = {}
      for layer_name in layers_cur_iteration:
        layers_prev[layer_name] = _LayerStateHelper(layer_name, "state")

    # We are ignoring acc_tas (the tensor arrays).

    self._set_construction_state_in_loop()

    # Create read ops now to maybe use them in the conditional branch.
    for layer_name in layers_cur_iteration:
      layers_prev[layer_name].get_reads_once()
    for layer_name in layers_delayed.union(layers_delayed_prev_deps):
      layers_prev_prev[layer_name].get_reads_once()

    with tf.name_scope("delayed_state_update"):
      # noinspection PyShadowingNames
      def _delayed_state_update():
        self._construct_custom(
          net=self.net_delayed_update,
          prev_state={
            layer_name: layers_prev_prev[layer_name].get_reads_once()
            for layer_name in layers_delayed.union(layers_delayed_prev_deps)},
          cur_state={
            layer_name: layers_prev[layer_name].get_reads_once()
            for layer_name in layers_cur_iteration},
          needed_outputs=layers_delayed)

        # Make sure all state vars dependencies are read first before we reassign them.
        control_deps = []
        for layer_name in layers_delayed.union(layers_delayed_prev_deps):
          control_deps += nest.flatten(layers_prev_prev[layer_name].get_reads_once())
        for layer_name in choice_layers:
          control_deps.append(self.net_delayed_update.layers[layer_name].output.placeholder)

        # Now reassign the delayed state vars.
        with tf.control_dependencies(control_deps):
          ops = []
          # Update all delayed state vars, i.e. prev:prev:layer to prev:layer.
          for layer_name in layers_delayed.union(layers_delayed_prev_deps):
            if layer_name in layers_cur_iteration:
              # Can simply copy it. "prev:layer" is already in it.
              state = layers_prev[layer_name].get_reads_once()
            else:
              assert layer_name in layers_delayed
              state = _LayerStateHelper.get_from_net(self.net_delayed_update, layer_name)
            ops += layers_prev_prev[layer_name].assigns_flat(state)

        return tf.group(*ops)

      self.delayed_state_update_op = tf.cond(tf.greater(i, initial_i), _delayed_state_update, tf.no_op)

      # We should not use control dependencies on delayed_state_update_op
      # because delayed_state_update_op will be update_ops and therefore executed first.

      # Now we should update net_vars accordingly

      for v in layers_prev_prev.values():
        v.reset_reads_once()
        v.get_reads_once()

      outputs_flat = []
      for layer_name in sorted(self._initial_outputs):
        if layer_name in layers_cur_iteration:
          outputs_flat.append(layers_prev[layer_name].get_reads_once()["output"])
        else:
          assert layer_name in layers_delayed
          outputs_flat.append(layers_prev_prev[layer_name].get_reads_once()["output"])  # This is updated now.
      extra_flat = []
      for layer_name, v in sorted(self._initial_extra_outputs.items()):
        if layer_name in layers_cur_iteration:
          state = layers_prev[layer_name].get_reads_once()["extra"]
        else:
          assert layer_name in layers_delayed
          state = layers_prev_prev[layer_name].get_reads_once()["extra"]  # This is updated now.
        assert isinstance(v, dict) and isinstance(state, dict)
        assert set(state.keys()) == set(v.keys())
        extra_flat.append(util.sorted_values_from_dict(state))
      net_vars = (outputs_flat, extra_flat)

    state_update_ops = []
    with tf.name_scope("cond"):
      s = rec_layer.create_state_var("cond", tf.constant(True))
      state_update_ops.append(s.assign(cond(i, net_vars, acc_tas, seq_len_info, allow_inf_max_len=True)))

    with tf.name_scope("body"):
      # The body function is locally defined in _SubnetworkRecCell.get_output().
      # This function then calls _SubnetworkRecCell._construct() which will do the subnet construction.
      res = body(i, net_vars, acc_tas, seq_len_info)
      assert len(res) == len(loop_vars)
      if len(res) == 3:
        i, net_vars, acc_tas = res
        seq_len_info = None
      else:
        i, net_vars, acc_tas, seq_len_info = res
    if seq_len_info:
      state_update_ops += rec_layer.assign_state_vars_recursive_flatten(("end_flag", "dyn_seq_len"), seq_len_info)
    if rec_layer.rec_step_by_step_opts["update_i_in_graph"]:
      state_update_ops.append(rec_layer.state_vars["i"].assign(i))

    # Assign new state.
    for layer_name in layers_cur_iteration:
      state = _LayerStateHelper.get_from_net(self.net, layer_name)
      state_update_ops += layers_prev[layer_name].assigns_flat(state)
    self.state_update_op = tf.group(*state_update_ops)

    return res

  def _construct_custom(self, net, prev_state, cur_state, needed_outputs):
    """
    This is a simplified version of _SubnetworkRecCell._construct without search logic.

    :param TFNetwork net:
    :param prev_state:
    :param cur_state:
    :param needed_outputs:
    """
    assert isinstance(net, TFNetwork)

    prev_layers = {}  # type: typing.Dict[str,_TemplateLayer]

    # noinspection PyShadowingNames
    def _add_predefined_layer(layer_name, state_dict, prev):
      assert isinstance(state_dict, dict)
      assert set(state_dict.keys()).issubset({"output", "extra"})
      try:
        layer = self.layer_data_templates[name].copy_as_prev_time_frame(
          prev_output=state_dict.get("output", None),
          rec_vars_prev_outputs=state_dict.get("extra", None))
      except Exception as exc:  # Some sanity check might have failed or so.
        self._handle_construct_exception(description="in-loop init of prev layer %r" % name, exception=exc)
        raise
      layer.network = net
      if prev:
        prev_layers[layer_name] = layer
      net.layers[("prev:%s" % layer_name) if prev else layer_name] = layer

    for name, state_dict in prev_state.items():
      _add_predefined_layer(name, state_dict, prev=True)
    for name, state_dict in cur_state.items():
      _add_predefined_layer(name, state_dict, prev=False)

    # noinspection PyShadowingNames
    def get_layer(name):
      """
      :param str name: layer name
      :rtype: LayerBase
      """
      if name.startswith("prev:"):
        return prev_layers[name[len("prev:"):]]
      if name.startswith("base:"):
        return self._get_parent_layer(name[len("base:"):])
      # noinspection PyBroadException
      try:
        layer = net.construct_layer(self.net_dict, name=name, get_layer=get_layer)
        if name == "end":
          # Special logic for the end layer, to always logical_or to the prev:end layer.
          assert layer.output.shape == (), "%s: 'end' layer %r unexpected shape" % (self.parent_rec_layer, layer)
          prev_end_layer = net.layers["prev:end"]
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
    for layer_name in sorted(needed_outputs):
      get_layer(layer_name)

  def _construct(self, prev_outputs, prev_extra, i, data=None,
                 inputs_moved_out_tas=None, needed_outputs=("output",)):
    """
    This is called from within the body of the while loop
    (`tf.while_loop` in :class:`RecLayer` but here only a single step)
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

    # Fixup all base state var dim tags in the template layers.
    # These templates are used to define prev layers,
    # e.g. like prev accum att weights, so we might need the parent dyn sizes now.
    all_dyn_parent_dim_tags = set()
    for layer in self.parent_net.get_root_network().get_all_layers_deep():
      if self.net in layer.network.get_network_hierarchy():
        continue
      all_dyn_parent_dim_tags.update([
        dim for dim in layer.output.dim_tags
        if dim.dimension is None and not dim.is_batch_dim()])
    for layer in self.layer_data_templates.values():
      assert isinstance(layer, _TemplateLayer)
      dim_tags = list(layer.output.dim_tags)
      for _i, tag in enumerate(dim_tags):
        if tag in all_dyn_parent_dim_tags:
          dim_tags[_i] = self.get_parent_dim_tag(tag)
      layer.output = layer.output.copy_template_new_dim_tags(dim_tags)
      layer.kwargs["output"] = layer.output

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

  RASR does the following logic (currently only a single stochastic var supported)::

    # initial state vars
    encode_ops(input_placeholders=...)  # -> assign (base and loop) state vars

    # state_vars are only the loop state vars here, not the base state vars.
    # Store initial loop state values.
    # Note that this is not really necessary always (including the logic described here)
    # because we anyway then assign them again in the first iteration of the decoder loop
    # and never need them otherwise again.
    state_vars.readout()

    # Main decoder loop
    while seq_not_ended(...):

      for hyp in current_hyps:  # (in practice, this is partially batched)

        state_vars.assign(...)  # for current hyp (might be skipped in first decoder loop iteration)
        decoder_input_vars.assign(...)  # for current hyp, the previous choice

        update_ops(decoder_input_vars)  # -> assign/update potentially some loop state vars

        # Assume only a single decode_ops here (single ChoiceLayer).
        # decoder_input_vars contains prev choices
        decode_ops(state_vars, decoder_input_vars)  # -> assign decoder_output_vars (scores)

        # note that decoder_input_vars for current hyp is still the previous (!) choice

        post_update_ops(state_vars, decoder_input_vars)  # -> assign new state_vars

        state_vars.readout()  # readout and store them for this hyp

      < select new hyps and prune some away, based on scores >

  Because RASR is the main application, we adopt the recurrence logic of the RecLayer to be compatible to RASR.

  Note that the update_ops, decode_ops and post_update_ops all depend on the previous state vars (obviously)
  or maybe decode_ops would depend on some updated state vars, updated by update_ops,
  or maybe post_update_ops would depend on some updated state vars, updated by decode_ops or update_ops.
  The logic flow is as outlined above.
  And all ops depends on the previous (!) choice vars and not on the current choice vars.
  This is in contrast to RETURNN, where the final loop state vars are potentially based on the current choice var.

  One way to solve this is by delaying the state vars by one iteration.
  So when "prev state vars" would normally refer to "prev:layer" (also including hidden state),
  now it would refer to "prev:prev:layer".
  Then for the decode_ops, we need to make sure first that we transfer "prev:prev:layer" to "prev:layer"
  and then we can calculate it as normal.
  Even when we merge the post_update_ops with the decode_ops, in many models,
  this still requires to do a lot (most) computation redundantly twice
  because for computing the decode_ops, we do most of the same computation which we would do for computing
  "layer" based on "prev:layer", but "layer" is not used here because we delayed the state vars by one iteration.

  We might keep all state vars twice, once for the current iteration (when possible),
  and once for the delayed iteration (when needed).
  But this is bad because this would cause a lot of overhead when the decoder needs to copy around the state vars.
  This is the case for RASR which handles high number of hypotheses, where the state vars are stored in CPU memory.

  As mentioned before, decode_ops shares a lot of the computation with updating "layer" based on "prev:layer".
  Or said differently, the calculation of "layer" is often not dependent on the current choice var.
  When we can figure out the dependencies exactly, we can know which state vars can stay in the current iteration,
  and which need to be delayed.
  There are potentially also some state vars which we need to keep both for the current and previous iteration
  in order to be able to compute the delayed update when this depends on it.

  So, for the loop state vars, we have two cases:

  - It refers to the current iteration.
  - It refers to the previous iteration.

  For some layer (which is part of the state because it is accessed via "prev:layer" or has hidden state),
  we can have three cases:

  - Loop state vars referring to the current iteration.
  - Loop state vars referring to the previous iteration.
  - Loop state vars for both the current and previous iteration.

  The update of delayed state vars must take extra care in the first frame.
  It is initialized with the initial state (layers initial_output, and initial hidden state),
  and in the first frame, it would just not update this.
  Only from the second frame on, it is delayed.

  How to we implement this logic here?

  First we aggressively search for layers which can stay in the current iteration,
  i.e. which do not depend on any of the :class:`ChoiceLayer`s.
  The update calculation from "prev:layer" to "layer" would be jointly with decode_ops
  to avoid redundant computation.
  This should be the first decode_ops if there are multiple.
  This makes the post_update_ops obsolete.

  For delayed state vars, before we do anything else,
  in the beginning of the iteration, we need to update "prev:prev:layer" to "prev:layer",
  or also set "prev:layer" based on the prev choices (e.g. "prev:output").
  For this, we need to know which dependencies on the state are needed for "layer" ("prev:layer"),
  and then all such dependencies are needed to be stored as delayed state vars as well.
  Then everything else in the iteration can be done as usual.
  This first step will be the update_ops.

  So first we need to figure out which state vars are needed of what kind.
  Then we must construct the update_ops for one part of layers.
  Then we must construct the decode_ops (including merged post_update_ops) for another set of layers,
  and there can be overlap both in the inputs and outputs,
  so these must be two separate constructions.

  Note that RASR only needs to know about the loop state vars, not the base state vars,
  and the stochastic state vars are also handled separately.
  RASR will re-assign and readout the loop state vars in every loop iteration.

  ---

  If you want to implement beam search in an external application which uses the compiled graph,
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

  ---

  This layer class derives from :class:`RecLayer` and adopts the logic to be able
  to construct the mentioned ops for step-by-step execution.

  The final ops are constructed in :func:`post_compile`
  and put into corresponding TF graph collections and the info json.

  However, the main construction logic happens before, in :func:`__init__`,
  and then further in :func:`SubnetworkRecCellSingleStep._while_loop`.

  ---

  See https://github.com/rwth-i6/returnn/pull/874 for some discussion.
  """

  layer_class = "rec_step_by_step"
  SubnetworkRecCell = SubnetworkRecCellSingleStep

  class ConstructionState:
    """construction states"""
    GetSources = Entity("get_sources")
    Init = Entity("init")
    InLoop = Entity("in_body")

  @classmethod
  def prepare_compile(cls, rec_layer_name, net_dict, opts):
    """
    :param str rec_layer_name:
    :param dict[str,dict[str]] net_dict:
    :param dict[str] opts:
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
    rec_layer_dict["rec_step_by_step_opts"] = opts

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
    global_vars_coll = tf_compat.v1.get_collection_ref("global_vars")

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
      if name == rec_layer.stochastic_var_order[-1]:
        assert choice_layer.score_dependent
      if choice_layer.score_dependent:
        calc_scores_op = rec_layer.state_vars["stochastic_var_scores_%s" % name].final_op()
        info["stochastic_vars"][name] = {
          "calc_scores_op": calc_scores_op.name,
          "scores_state_var": "stochastic_var_scores_%s" % name,
          "choice_state_var": "stochastic_var_choice_%s" % name}
        if name == rec_layer.stochastic_var_order[-1]:
          # The last decode op can be merged with the state updates (post_update_ops).
          calc_scores_op = tf.group(calc_scores_op, cell.state_update_op)
        decode_ops_coll.append(calc_scores_op)
        decoder_output_vars_coll.append(rec_layer.state_vars["stochastic_var_scores_%s" % name].var)
      else:
        info["stochastic_vars"][name] = {"choice_state_var": "stochastic_var_choice_%s" % name}
      decoder_input_vars_coll.append(rec_layer.state_vars["stochastic_var_choice_%s" % name].var)

    update_ops_coll = tf_compat.v1.get_collection_ref("update_ops")
    update_ops_coll.append(cell.delayed_state_update_op)
    # Based on the decoder state, encoder, and choices, calculate the next state.
    tf_compat.v1.get_collection_ref("post_update_ops")  # leave empty; merged with last decode op
    state_vars_coll = tf_compat.v1.get_collection_ref(CollectionKeys.STATE_VARS)
    for name, var in sorted(rec_layer.state_vars.items()):
      assert isinstance(name, str)
      assert isinstance(var, RecStepByStepLayer.StateVar)
      if not name.startswith("stochastic_var_") and not name.startswith("base_") and name != "cond":
        if var.orig_data_shape.have_batch_axis():
          state_vars_coll.append(var.var)
        else:
          global_vars_coll.append(var.var)

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
      zero_initializer = tf_util.zeros_dyn_shape(
        shape=self.var_data_shape.batch_shape, dtype=self.var_data_shape.dtype)
      with helper_variable_scope():
        self.var = tf_compat.v1.get_variable(
          name=tf_util.get_valid_scope_name_from_str(name),
          initializer=zero_initializer,
          validate_shape=False)  # type: tf.Variable
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
      return self.assign(self.final_value)

    def assign(self, value):
      """
      :return: op which does self.var.assign(value) (value maybe converted)
      :rtype: tf.Operation
      """
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

  def __init__(self, _orig_sources, sources, network, name, output, unit, axis, rec_step_by_step_opts,
               reverse_stochastic_var_order=False, **kwargs):
    """
    :param str|list[str]|None _orig_sources:
    :param list[LayerBase] sources:
    :param returnn.tf.network.TFNetwork network:
    :param str name:
    :param Data output:
    :param SubnetworkRecCellSingleStep unit:
    :param Dim axis:
    :param dict[str] rec_step_by_step_opts:
    :param bool reverse_stochastic_var_order:
    """
    assert isinstance(unit, SubnetworkRecCellSingleStep)
    kwargs = kwargs.copy()
    kwargs["optimize_move_layers_out"] = False
    self.state_vars = {}  # type: typing.Dict[str,RecStepByStepLayer.StateVar]
    self.stochastic_var_order = []  # type: typing.List[str]
    self.stochastic_vars = {}  # type: typing.Dict[str,RecStepByStepLayer.StochasticVar]
    self._parent_tile_multiples = None  # type: typing.Optional[tf.Tensor]
    self.construction_state = self.ConstructionState.GetSources
    self.reverse_stochastic_var_order = reverse_stochastic_var_order
    self.rec_step_by_step_opts = rec_step_by_step_opts

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
    # This will eventually get to SubnetworkRecCellSingleStep._while_loop.
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
    :rtype: RecStepByStepLayer.StateVar
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
    return var

  def set_state_var_final_value(self, name, final_value):
    """
    :param str name:
    :param tf.Tensor final_value:
    """
    self.state_vars[name].set_final_value(final_value)

  def create_state_vars_recursive(self, name_prefix, initial_values, data_shapes=None):
    """
    :param str|tuple[str] name_prefix: single or same structure as initial_values
    :param T initial_values:
    :param data_shapes: same structure as initial_values or None, but values are of instance :class:`Data`
    :return: same as initial_values, but the state vars
    """
    def _map_state_vars(path, name, initial_value, data_shape):
      assert isinstance(initial_value, tf.Tensor)
      assert data_shape is None or isinstance(data_shape, Data)
      if not name:
        name = "/".join(map(str, (name_prefix,) + path))
      return self.create_state_var(name=name, initial_value=initial_value, data_shape=data_shape)

    if isinstance(name_prefix, str):
      name_per_entry = nest.map_structure(lambda v: None, initial_values)
    else:
      name_per_entry = name_prefix
    if data_shapes is None:
      data_shapes = nest.map_structure(lambda v: None, initial_values)
    nest.assert_same_structure(initial_values, name_per_entry)
    nest.assert_same_structure(initial_values, data_shapes)
    return nest.map_structure_with_tuple_paths(
      _map_state_vars, name_per_entry, initial_values, data_shapes)

  def assign_state_vars_recursive_flatten(self, name_prefix, values):
    """
    :param str|tuple[str] name_prefix:
    :param T values:
    :rtype: list[tf.Operation]
    """
    def _map_state_vars(path, name, value):
      assert isinstance(value, tf.Tensor)
      if not name:
        name = "/".join(map(str, (name_prefix,) + path))
      return self.state_vars[name].assign(value)

    if isinstance(name_prefix, str):
      name_per_entry = nest.map_structure(lambda v: None, values)
    else:
      name_per_entry = name_prefix
    nest.assert_same_structure(values, name_per_entry)
    return nest.flatten(nest.map_structure_with_tuple_paths(
      _map_state_vars, name_per_entry, values))

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

  class StochasticVar:
    """
    Manages a stochastic variable, which corresponds to a :class:`ChoiceLayer`.
    """
    def __init__(self, parent_rec_layer, layer_template):
      """
      :param RecStepByStepLayer parent_rec_layer:
      :param _TemplateLayer layer_template:
      """
      self.parent_rec_layer = parent_rec_layer
      self.name = layer_template.name
      self.choice_layer_opts = layer_template.kwargs
      # for compatibility: one way to set the ngram label context via additional choices
      # but older history is not score-dependent stochastic_var anymore
      self.score_dependent = self.choice_layer_opts.get("score_dependent", True)
      self.score_state_var = None
      if self.score_dependent:
        sources = layer_template.kwargs["sources"]
        assert len(sources) == 1
        source_template, = sources
        self.score_state_var = self.parent_rec_layer.create_state_var(
          name="stochastic_var_scores_%s" % self.name, data_shape=source_template.output)
      self.choice_state_var = self.parent_rec_layer.create_state_var(
        name="stochastic_var_choice_%s" % self.name, data_shape=layer_template.output)

    def assign_score(self, source):
      """
      :param LayerBase source:
      """
      assert self.score_dependent
      assert source.output.is_batch_major and len(source.output.shape) == 1
      scores_in = source.output.placeholder
      input_type = self.choice_layer_opts.get("input_type", "prob")
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
      self.score_state_var.set_final_value(final_value=scores_in)

    def get_choice(self):
      """
      :return: the choice value
      :rtype: tf.Tensor
      """
      return self.choice_state_var.read()

  def add_stochastic_var(self, name):
    """
    :param str name:
    :rtype: RecStepByStepLayer.StochasticVar
    """
    assert name not in self.stochastic_vars
    cell = self.cell
    assert isinstance(cell, SubnetworkRecCellSingleStep)
    self.stochastic_vars[name] = self.StochasticVar(self, cell.layer_data_templates[name])
    return self.stochastic_vars[name]

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
    self.stochastic_var = rec_layer.stochastic_vars[self.name]
    cell = rec_layer.cell
    assert isinstance(cell, SubnetworkRecCellSingleStep)
    assert self.network in {cell.net, cell.net_delayed_update}
    if self.network == cell.net:
      rec_layer.stochastic_var_order.append(self.name)
    # for compatibility: one way to set the ngram label context via additional choices
    # but older history is not score-dependent stochastic_var anymore
    self.score_dependent = score_dependent
    if self.score_dependent:
      assert len(self.sources) == 1
      source, = self.sources
      self.stochastic_var.assign_score(source)
    self.output.placeholder = self.stochastic_var.get_choice()

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
    parent_rec_layer = network.parent_layer
    if parent_rec_layer:  # might be None at first template construction
      assert isinstance(parent_rec_layer, RecStepByStepLayer)
      cell = parent_rec_layer.cell
      assert isinstance(cell, SubnetworkRecCellSingleStep)
      assert network in {cell.net, cell.net_delayed_update}
      if network == cell.net_delayed_update:
        # Remove the dependency to the scores.
        d["from"] = ()
        d["score_dependent"] = False
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
  argparser.add_argument(
    '--device', default="cpu", help="'cpu' (default) or 'gpu'. optimizes the graph for this device")
  argparser.add_argument("--verbosity", default=4, type=int, help="5 for all seqs (default: 4)")
  argparser.add_argument("--summaries_tensor_name", help="create Tensor for tf.compat.v1.summary.merge_all()")
  argparser.add_argument("--rec_step_by_step", help="make step-by-step graph for this rec layer (eg. 'output')")
  argparser.add_argument("--rec_step_by_step_output_file", help="store meta info for rec_step_by_step (JSON)")
  argparser.add_argument("--output_file", help='allowed extensions: pb, pbtxt, meta, metatxt, logdir')
  argparser.add_argument("--output_file_model_params_list", help="line-based, names of model params")
  argparser.add_argument("--output_file_state_vars_list", help="line-based, name of state vars")
  argparser.add_argument("--update_i_in_graph", action="store_true", help="whether to update i in the graph")
  args = argparser.parse_args(argv[1:])
  assert args.train in [0, 1, -1] and args.eval in [0, 1] and args.search in [0, 1]
  init(config_filename=args.config, log_verbosity=args.verbosity, device=args.device)
  assert 'network' in config.typed_dict
  net_dict = config.typed_dict["network"]
  if args.rec_step_by_step:
    RecStepByStepLayer.prepare_compile(
      rec_layer_name=args.rec_step_by_step, net_dict=net_dict, opts={"update_i_in_graph": args.update_i_in_graph})
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
        out = layer.output.copy_as_batch_major()
        if out.have_feature_axis():
          out = out.copy_with_feature_last()
        tf.identity(out.placeholder, name="output_batch_major")

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
