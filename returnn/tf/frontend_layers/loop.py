"""
Loop. Provides :class:`Loop`.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union, Iterable
import tree
from returnn.util.basic import NotSpecified
from returnn.tensor import Tensor, Dim, ControlFlowContext
import returnn.frontend as rf
import returnn.tf.frontend_layers as rfl
from . import _utils


__all__ = ["Loop", "LoopModule"]


class Loop:
    """
    This represents a RecLayer subnetwork in RETURNN,
    i.e. where the calculation per step is defined explicitly.

    (For RecLayer with a predefined unit, see :class:`Rec`.
     Or for example :class:`Lstm`.)

    To define a loop like this pseudo Python code::

      x  # given, shape (batch, time, dim)
      h = Zeros([batch,dim])()  # initial state, shape (batch,dim)
      out = []
      for t in range(x.max_seq_len):
        x_lin = Linear(dim)(x[t])
        h_prev = h
        h = Linear(dim)(x_lin + h_prev)
        out.append(h)

      h  # final state
      out  # shape (time, batch, h_dim)

    You would write::

      dim = nn.FeatureDim(...)
      loop = nn.Loop(axis=...)
      loop.state.h = nn.zeros([batch_dim,dim])  # initial state
      with loop:
        x_t = loop.unstack(x)
        x_lin = Linear(dim)(x_t)
        loop.state.h = Linear(dim)(x_lin + loop.state.h)
        out = loop.stack(loop.state.h)

    ``state`` is :class:`Loop._StateHolder` and manages the recurrent state.

    This code must be run within a :func:`Module.forward`
    or with some active global name context (:class:`NameCtx`).

    This API is currently in development, and might change.
    See: https://github.com/rwth-i6/returnn_common/issues/16
    """

    def __init__(
        self,
        *,
        max_seq_len: Optional[Tensor] = NotSpecified,
        optimize_move_layers_out: Optional[bool] = NotSpecified,
        unroll: bool = NotSpecified,
        axis: Optional[Dim] = NotSpecified,
        debug: Optional[bool] = NotSpecified,
        name: str = "loop",
    ):
        super(Loop, self).__init__()
        self._has_given_axis = True
        if not axis or axis is NotSpecified:
            self._has_given_axis = False
            axis = Dim(None, name=f"{name}-dim")
        assert isinstance(axis, Dim)
        self.extra_opts = {
            {"max_seq_len": "max_seq_len_via"}.get(key, key): value
            for (key, value) in locals().items()
            if value is not NotSpecified and value is not None and key not in {"self", "__class__", "name"}
        }
        self.layer_module = LoopModule(loop=self)
        parent_name_ctx = rfl.Layer.current_ctx()
        self.control_flow_ctx = ControlFlowContext(
            kind=ControlFlowContext.Types.Loop,
            outer_ctx=rfl.Layer.inner_control_flow(),
            identifier=parent_name_ctx.get_abs_name(),
        )
        self.control_flow_ctx.loop_spatial_dim = axis
        self.name_ctx = rfl.Layer(
            module=self.layer_module,
            suggested_name=name,
            parent=parent_name_ctx,
            new_control_flow_ctx=self.control_flow_ctx,
            can_access_children=False,
        )
        self.name_ctx.custom_layer_name_scope = ""
        self.name_ctx.is_subnet = True
        self.name_ctx.extend_reserved_names({"output", "end"})
        self._entered_scope = False
        self._exited_scope = False
        self._exited_scope_with_exception = False
        self._state = _LoopStateHolder(loop=self)
        self.unstacked_refs = []  # type: List[Tensor]
        self.outputs = []  # type: List[Tensor]
        self._last_frames = {}  # type: Dict[rfl.Layer, Tensor]  # inner name -> outer
        self.axis = axis
        self.end_ref = None  # type: Optional[Tensor]
        self._iter_idx_ref = None  # type: Optional[Tensor]

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name_ctx.get_abs_name_repr()}>"

    def __enter__(self) -> Loop:
        assert not self._entered_scope, f"{self}: cannot enter twice"
        self._entered_scope = True
        self.name_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert not self._exited_scope, f"{self}: cannot exit twice"
        try:
            if not exc_type:
                if self.end_ref is None and not self.unstacked_refs:
                    raise Exception(f"{self}: call `unstack` or `end` at least once to define the loop length")
                # Make sure there is an "output" layer. (Similar as for Module with subnetwork.)
                if not self.outputs:
                    # Stack some dummy output
                    if self.end_ref is not None:
                        self.stack(self.end_ref)
                    else:
                        assert self.unstacked_refs
                        self.stack(self.unstacked_refs[0])
                assert "output" in self.name_ctx.children
        finally:
            self._exited_scope_with_exception = bool(exc_type)
            self._exited_scope = True
            self.name_ctx.__exit__(exc_type, exc_val, exc_tb)
        if not exc_type:
            res = self.layer_module()  # create the rec layer itself
            if self.end_ref is not None:
                res.raw_tensor.layer_extra_dependencies.append(self.end_ref.raw_tensor)

    @property
    def has_entered_scope(self) -> bool:
        """
        :return: whether we have entered the scope, i.e. we define the per-step calculation.
        """
        return self._entered_scope

    @property
    def state(self) -> Union[_LoopStateHolder, rf.State]:
        """state holder inside the loop"""
        if not self._exited_scope:
            return self._state
        if self._exited_scope_with_exception:  # nicer for debugging
            return self._state
        # noinspection PyProtectedMember
        return self._state._get_last()

    @state.setter
    def state(self, initial_state: rf.State):
        assert len(self._state) == 0, f"can only assign {self}.state once for the initial state"
        assert not self._entered_scope
        for key, value in initial_state.items():
            self._state[key] = value

    def unstack(self, source: Tensor, *, name: Optional[str] = None) -> Tensor:
        """
        Unrolls over the specified axis, and provides each frame in each loop iteration.
        The axis can be specified globally for the :class:`Loop` instance (recommended)
        or locally here (not recommended).
        """
        assert self._has_given_axis, "%s: unstack() requires a given axis" % self
        assert self.axis in source.dims
        res = _rec_unstack(source, axis=self.axis, name=name)
        self.unstacked_refs.append(res)
        return res

    def stack(self, source: Tensor, *, name: Optional[str] = None) -> Tensor:
        """
        Accumulates the frames of source within the loop,
        to make it accessible outside the loop.
        """
        # We don't need to do anything special because RETURNN RecLayer will automatically accumulate the frames
        # when we marked a layer with is_output_layer, and we access it from outside the loop.
        if not name and "output" not in self.name_ctx.children:
            name = self.name_ctx.get_child("output")
        if isinstance(name, str) or not name:
            if not name:
                name = source.name
            name = rfl.Layer(suggested_name=name, parent=self.name_ctx)
        assert isinstance(name, rfl.Layer)

        # We access the returned layer-ref from outside, thus fix the data template.
        data = source.copy_template().copy_add_dim_by_tag(dim_tag=self.axis, unbroadcast=True, axis=0)
        data.time_dim_axis = 0
        data.control_flow_ctx = self.control_flow_ctx.outer_ctx
        # Because the self.axis dim is yet unknown,
        # the logic of _register_dim_deps_when_novel will create a length layer
        # to make the length available.
        # However, this length layer will be inside the rec loop, which we do not want...
        with self.name_ctx.parent:
            res = rfl.make_layer({"class": "copy", "from": source}, predefined_out_data=data, name=name)
        assert isinstance(res, Tensor)
        if res.raw_tensor.name != "output":
            res.raw_tensor.layer_dict["is_output_layer"] = True
        self.outputs.append(res)
        return res

    def last(self, source: Tensor, *, name: Optional[str] = None) -> Tensor:
        """
        Gets the last value from source.
        """
        assert isinstance(source, Tensor)
        if source.raw_tensor in self._last_frames:
            return self._last_frames[source.raw_tensor]
        assert self.name_ctx.tensor is not None, f"{self}.last(...): must call from outside"  # current restriction...
        # need_last option only works in root of subnet of RecLayer
        if source.raw_tensor.parent is not self.name_ctx:
            assert self.name_ctx in source.raw_tensor.get_abs_name_ctx_list(), f"invalid {self}.last({source})"
            sub_layer_name = source.raw_tensor.get_name_in_ctx(self.name_ctx).replace("/", ".")
            source = _utils.copy(source, name=self.name_ctx.get_new_child(sub_layer_name))
            assert source.raw_tensor.parent is self.name_ctx
        source.raw_tensor.layer_dict["need_last"] = True
        sub_layer_name = source.raw_tensor.get_name_in_ctx(self.name_ctx)
        with self.name_ctx.parent:  # need to be outside the loop
            res = rfl.make_layer(
                {"class": "rec_last_output", "rec_layer": self.name_ctx.tensor, "sub_layer_name": sub_layer_name},
                predefined_out_data=source,
                name=name or sub_layer_name.replace("/", "_"),
            )
            res.raw_tensor.tensor_remove_unused_cleanup_hooks.append(
                lambda _: source.raw_tensor.layer_dict.pop("need_last")
            )
            res.raw_tensor.layer_extra_dependencies.append(source.raw_tensor)
            source.raw_tensor.usages.append(res.raw_tensor)
            self._last_frames[source.raw_tensor] = res
            return res

    def end(self, source: Tensor, *, include_eos: bool) -> Tensor:
        """
        For loops with dynamic ending condition (which might not use unstack),
        this defines the ending condition.

        :param source: the ending condition
        :param include_eos: if True, the last() and stack() function include the current ending frame, otherwise not
        """
        assert not self.end_ref, f"{self}.end() can only be called once"
        assert source.dtype == "bool", f"{self}: end expects boolean condition, got {source}"
        if not self.axis.dyn_size_ext:
            dyn_size_ext = source.copy_template()
            dyn_size_ext.dtype = "int32"
            if dyn_size_ext.control_flow_ctx:
                dyn_size_ext.control_flow_ctx = dyn_size_ext.control_flow_ctx.outer_ctx
            self.axis.dyn_size_ext = dyn_size_ext
            self.axis.batch = dyn_size_ext.batch
            self.axis.control_flow_ctx = dyn_size_ext.control_flow_ctx
        self.extra_opts["include_eos"] = include_eos
        self.end_ref = _utils.copy(source, name=self.name_ctx.get_child("end"))
        return self.end_ref

    @property
    def max_seq_len(self) -> Optional[Tensor]:
        """max seq length in case the length is dynamic via :func:`end`"""
        return self.extra_opts.get("max_seq_len_via")

    @max_seq_len.setter
    def max_seq_len(self, value: Optional[Tensor]):
        if value is None:
            self.extra_opts.pop("max_seq_len_via", None)
        else:
            self.extra_opts["max_seq_len_via"] = value

    @property
    def iter_idx(self) -> Tensor:
        """
        The index of the current iteration, inside the loop. This is a scalar. This always starts with 0.

        """
        assert self._entered_scope and not self._exited_scope
        if self._iter_idx_ref is not None:
            return self._iter_idx_ref
        self._iter_idx_ref = self.name_ctx.get_child_tensor(
            ":i",
            data=Tensor(":i", dtype="int32", dim_tags=(), sparse_dim=self.axis, control_flow_ctx=self.control_flow_ctx),
        )
        return self._iter_idx_ref


class LoopModule(rf.Module):
    """
    This module is used internally by :class:`Loop` to create the RETURNN :class:`RecLayer` for the loop.
    This module would not be directly used by the user.
    """

    def __init__(self, loop: Loop):
        super(LoopModule, self).__init__()
        self.loop = loop

    def __call__(self) -> Tensor:
        """
        Makes layer dict for this loop, i.e. a RecLayer.
        """
        name_ctx = self.loop.name_ctx
        out = name_ctx.children["output"].tensor
        # self.stack already added the loop.axis dim tag to prepare the access from outside the loop.
        assert out.dim_tags[0] == self.loop.axis
        return rfl.make_layer(
            {"class": "rec", "from": [], "unit": name_ctx.make_net(), **self.loop.extra_opts},
            name=name_ctx,
            predefined_out_data=out,
        )


class _LoopStateHolder:
    def __init__(self, loop: Loop):
        self._loop = loop
        self._state = {}  # type: Dict[str, _LoopState]

    def __repr__(self):
        return f"{self._loop}.state"

    def _get_state(self, name: str) -> _LoopState:
        if name in self._state:
            return self._state[name]
        raise AttributeError(f"{self}: Unknown state attrib {name!r}. Assign the initial state first.")

    def _get_last(self) -> rf.State:
        return rf.State({key: value.get_last() for (key, value) in self._state.items()})

    def __getitem__(self, item):
        return self._get_state(item).get()

    def __setitem__(self, key, value):
        if not self._loop.has_entered_scope:
            assert key not in self._state, f"{self} already has state {key!r}"
            self._state[key] = _LoopState(name=key, loop=self._loop, initial=value)
            return
        self._get_state(key).assign(value)

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        if key in {"_state", "_loop"}:
            return super().__setattr__(key, value)
        self[key] = value

    def keys(self) -> Iterable[str]:
        """keys"""
        return self._state.keys()

    def values(self) -> List[Any]:
        """values"""
        return [v.get() for v in self._state.values()]

    def __len__(self):
        return len(self._state)

    def deep_tensors(self) -> List[Tensor]:
        """See :func:`LayerState.cls_deep_tensors`."""
        return rf.State.cls_deep_tensors(self)


class _LoopState:
    """
    Represents some recurrent state, to be used with :class:`Loop`.
    It can also represent some nested hierarchy of states.
    """

    def __init__(self, *, name: str, loop: Loop, initial: Union[Tensor, Any]):
        """
        :param name:
        :param loop:
        :param initial: some layer-ref, or any kind of nested structure of layers.
        """
        super(_LoopState, self).__init__()
        assert initial is not None
        initial = tree.map_structure(rf.convert_to_tensor, initial)
        self.initial = initial
        self.loop = loop
        self.name = name
        self.assigned_value = None
        self.name_ctx = tree.map_structure_with_path(
            lambda path, ref: rfl.Layer(
                suggested_name=".".join(str(key) for key in ("state", name) + path), parent=loop.name_ctx
            ),
            self.initial,
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name!r}>"

    def assign(self, value):
        """
        Assign the new value of the current iteration.
        This is called (only) inside the loop.
        This will define the value for the next iteration.
        """
        assert self.name_ctx is not None
        assert value is not None
        assert self.assigned_value is None, (
            f"Cannot assign the rec state {self.loop}/{self.name} multiple times, "
            f"assigned previously to {self.assigned_value}, now to {value}"
        )
        tree.assert_same_structure(self.initial, value)
        tree.assert_same_structure(self.name_ctx, value)
        self.assigned_value = value

        def _map_ref_to_name_ctx(tensor: Tensor, name_ctx: rfl.Layer, initial: Tensor):
            assert isinstance(tensor, Tensor)
            assert isinstance(name_ctx, rfl.Layer)
            assert name_ctx.tensor is None, f"Loop state {name_ctx} already assigned"

            tensor.raw_tensor.make_all_sub_networks_and_optimize()

            layer_ctx_list = tensor.raw_tensor.get_abs_name_ctx_list()
            assert (
                self.loop.name_ctx in layer_ctx_list
            ), f"Loop state {name_ctx} should get a value inside the loop but got {tensor}"
            # We need some special logic for MaskedComputation but maybe also for others later.
            # This is currently not nice, but I'm not sure about better solutions.
            for i in range(layer_ctx_list.index(self.loop.name_ctx) + 1, len(layer_ctx_list) - 1):
                ctx, ctx_ = layer_ctx_list[i : i + 2]
                assert isinstance(ctx, rfl.Layer) and isinstance(ctx_, rfl.Layer)
                if isinstance(ctx.module, rfl.MaskedComputationModule):
                    ctx_.layer_dict["is_output_layer"] = True
                    break

            # Potential optimization for RETURNN layers.
            # See ReturnnWrappedLayerBase._get_recurrent_state.
            if tensor.raw_tensor.layer_dict:
                _do_const_initial_value_opt = False
                _const_initial_value_opt_layer_white_list = {"cum_concat", "rec"}
                if tensor.raw_tensor.layer_dict["class"] in _const_initial_value_opt_layer_white_list:
                    _do_const_initial_value_opt = True
                elif tensor.raw_tensor.layer_dict["class"] == "get_last_hidden_state":
                    src = tensor.raw_tensor.layer_dict["from"]
                    assert isinstance(src, Tensor)
                    if src.raw_tensor.layer_dict:
                        if src.raw_tensor.layer_dict["class"] in _const_initial_value_opt_layer_white_list:
                            _do_const_initial_value_opt = True
                if _do_const_initial_value_opt:
                    # Note: Only do this optimization for some layers because otherwise
                    # we might rely on the initial output shape.
                    initial_const = _utils.constant_value(initial)
                    if initial_const is not None:
                        initial = initial_const

                if tensor.raw_tensor.layer_dict["class"] == "get_last_hidden_state":
                    used_state_eliminate_optimization = False
                    key = tensor.raw_tensor.layer_dict.get("key", "state")
                    src = tensor.raw_tensor.layer_dict["from"]
                    assert isinstance(src, Tensor)
                    src_state_opt = src.raw_tensor.layer_dict.get("state") if src.raw_tensor.layer_dict else None
                    if isinstance(src_state_opt, rf.State):
                        src_state_for_key = src_state_opt.get(key)
                        if isinstance(src_state_for_key, rfl.PrevTensorRef):
                            if src_state_for_key.cur_layer_name_ctx is name_ctx:
                                # The 'state' argument of the rec layer refers to "prev:..." of the state.
                                # So we don't need to pass it now.
                                used_state_eliminate_optimization = True
                                src_state_opt[key] = None
                                if all(opt is None for opt in tree.flatten(src_state_opt)):
                                    del src.raw_tensor.layer_dict["state"]
                                # We need to pass the initial_state instead though.
                                src_initial_state_opt = src.raw_tensor.layer_dict.setdefault(
                                    "initial_state", rf.State()
                                )
                                src_initial_state_opt[key] = initial
                                # If there is any other code which refers to this state, it can access the passed layer.
                                # So anyway pass through.

                    if not used_state_eliminate_optimization:
                        raise NotImplementedError(
                            f"{self}.assign to {tensor} on {src}:"
                            f" We need https://github.com/rwth-i6/returnn_common/issues/31"
                            f" and https://github.com/rwth-i6/returnn/issues/732."
                        )

                else:  # class != get_last_hidden_state
                    if tensor.raw_tensor.layer_dict["class"] == "cum_concat":
                        layer_state_opt = tensor.raw_tensor.layer_dict.get("state")
                        if isinstance(layer_state_opt, rf.State) and set(layer_state_opt.keys()) == {"state"}:
                            layer_state = layer_state_opt.state
                            if (
                                isinstance(layer_state, rfl.PrevTensorRef)
                                and layer_state.cur_layer_name_ctx is name_ctx
                            ):
                                # The 'state' argument refers to "prev:..." of itself.
                                # This is redundant, so we don't need to pass it.
                                tensor.raw_tensor.layer_dict.pop("state")

                    assert "initial_state" not in tensor.raw_tensor.layer_dict
                    assert "initial_output" not in tensor.raw_tensor.layer_dict
                    tensor.raw_tensor.layer_dict["initial_output"] = initial

            else:  # tensor not Tensor
                raise NotImplementedError(f"{self}.assign to {tensor} (type {type(tensor)}) but Tensor expected")

            # Note: We assume this has been used before in get() -> PrevTensorRef.get_prev_ref().
            prev_name_ctx = name_ctx.parent.children.get(f"prev:{name_ctx.name}")
            if prev_name_ctx:  # might not exist if we have never accessed the prev state
                prev_ref = prev_name_ctx.tensor
                assert isinstance(prev_ref, rfl.PrevTensorRef), f"{name_ctx, prev_name_ctx}"
                if tensor.raw_tensor.parent != self.loop.name_ctx:
                    # Currently, RETURNN does not properly support a state in a subnet.
                    # So we copy the layer to the loop root under the reserved existing name.
                    _utils.copy(tensor, name=name_ctx)
                    if tensor.raw_tensor.layer_dict:
                        assert "initial_state" not in tensor.raw_tensor.layer_dict  # not supported/implemented
                        if "initial_output" in tensor.raw_tensor.layer_dict:
                            name_ctx.layer_dict["initial_output"] = tensor.raw_tensor.layer_dict.pop("initial_output")
                else:
                    prev_ref.assign_new_cur_tensor_name_ctx(tensor.raw_tensor)

            return tensor.raw_tensor

        self.name_ctx = tree.map_structure(_map_ref_to_name_ctx, value, self.name_ctx, self.initial)

    @staticmethod
    def _map_name_ctx_to_prev_tensor(name_ctx: rfl.Layer, initial: Tensor) -> rfl.PrevTensorRef:
        assert isinstance(name_ctx, rfl.Layer)
        return rfl.PrevTensorRef.get_prev_ref(cur_layer_name_ctx=name_ctx, initial=initial)

    def get(self):
        """
        Return prev or current value of the current loop iteration,
        depending on whether assign() already has been called or not.
        This is called (only) inside a loop.
        """
        assert self.name_ctx is not None
        if not self.loop.has_entered_scope:
            return self.initial
        if self.assigned_value is None:  # not yet assigned
            # Return prev value
            return tree.map_structure(self._map_name_ctx_to_prev_tensor, self.name_ctx, self.initial)
        return self.assigned_value

    def _map_name_ctx_to_last_tensor(self, name_ctx: rfl.Layer) -> Tensor:
        assert isinstance(name_ctx, rfl.Layer)
        assert name_ctx.tensor, f"{self.loop} state {name_ctx} not assigned?"
        assert self.loop.name_ctx.tensor, f"{self.loop} not yet exited?"
        return self.loop.last(name_ctx.tensor)

    def get_last(self):
        """
        Outside the loop, get the last instance.
        """
        assert self.name_ctx is not None
        assert self.assigned_value is not None
        return tree.map_structure(self._map_name_ctx_to_last_tensor, self.name_ctx)


# noinspection PyShadowingBuiltins,PyShadowingNames
def _rec_unstack(
    source: Tensor,
    *,
    axis: Dim,
    declare_rec_time: bool = NotSpecified,
    name: Optional[Union[str, rfl.Layer]] = None,
) -> Tensor:
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

    Due to automatic optimization, not much happens here.
    The real logic happens in :func:`get_out_data_from_opts`.

    Note that it is allowed to leave both `axis` and `declare_rec_time` unset,
    in case you assign `axis` to the rec layer, and the source here has the same axis (dim tag).

    :param nn.Tensor source:
    :param nn.Dim axis:
    :param bool declare_rec_time:
    :param str|nn.NameCtx|None name:
    :return: layer
    """
    if not isinstance(source, Tensor):
        raise TypeError(f"rec_unstack: unexpected type for source {source!r}, need tensor")
    args = {
        "axis": axis,
        "declare_rec_time": declare_rec_time,
    }
    args = {key: value for (key, value) in args.items() if value is not NotSpecified}
    return rfl.make_layer({"class": "rec_unstack", "from": source, **args}, name=name or "rec_unstack")
