"""
Layer (NameCtx earlier)
"""

from __future__ import annotations
from typing import TypeVar, Optional, Union, Any, Dict, Collection, Sequence, List, Tuple, Set, Callable
import types
import numpy
from tensorflow.python.util import nest
from returnn.util.basic import NotSpecified, RefIdEq
from returnn.tensor import Tensor, Dim, ControlFlowContext, batch_dim, single_step_dim
from returnn.tensor.marked_dim import MarkedDim as _MarkedDim
from returnn.tf.util.data import BatchInfo
from returnn.tf.util import basic as tf_util
from returnn.tf.layers.base import LayerBase
from .. import frontend_layers as rfl
from . import _utils
import returnn.frontend as rf


LayerDictRaw = Dict[str, Any]
NetDictRaw = Dict[str, LayerDictRaw]
T = TypeVar("T")


class Layer:
    """
    This is a helper class to keep track of the current name context when creating layers.
    Usually you do not need to access this directly
    except for creating the root name ctx
    and getting out the final RETURNN config or net dict.

    A name ctx represents one absolute layer name in the RETURNN layer hierarchy,
    except for the root name ctx.

    A name ctx thus can have a parent name ctx (if it is not the root),
    and potentially child name contexts.

    See the documentation on name hierarchies for RETURNN and RETURNN-common in the module docstring at the top.

    (Note: This class was previously called NameCtx in RETURNN-common.)
    """

    _stack = []  # type: List[Layer]
    _recent = None  # type: Optional[Layer]
    _ReservedNames = {"data", "output"}

    @classmethod
    def reset_default_root(cls):
        """
        Resets the default root name ctx.
        """
        cls._stack[0:1] = [cls.new_root()]
        cls._recent = None

    @classmethod
    def _maybe_init_default_root(cls):
        """
        Initialize the default root name ctx.
        """
        if not cls._stack:
            cls.reset_default_root()

    @classmethod
    def top(cls) -> Layer:
        """
        Return the top of the stack.
        Assumes that it exists.
        """
        cls._maybe_init_default_root()
        assert cls._stack
        return cls._stack[-1]

    @classmethod
    def recent_subnet(cls) -> Layer:
        """
        Return the most recent subnet.
        """
        top = cls.top()
        recent = cls._recent
        if recent and recent.root is top.root:
            while not recent.is_subnet:
                assert recent.parent
                recent = recent.parent
            assert recent.is_subnet
            return recent
        return top

    @classmethod
    def current_ctx(cls, *, ignore_top_stack_frames: int = 0) -> Layer:
        """
        Return the current context.
        This is the top from the stack with is_subnet_ctx,
        and additionally using the Python stack trace to automatically infer further subnets.

        :param int ignore_top_stack_frames:
        """
        return _auto_setup_parent_name_ctx(ignore_top_stack_frames=ignore_top_stack_frames + 1)

    @classmethod
    def new_root(cls) -> Layer:
        """
        Create new root name context
        """
        ctx = Layer(parent=None)
        ctx.is_subnet = True
        return ctx

    @classmethod
    def inner_loop(cls) -> Optional[ControlFlowContext]:
        """
        :return: the most inner loop in the current context, if there is one
          E.g. you can use it to access the outer spatial dim.
        """
        layer = cls.top()
        while layer:
            ctx = layer.new_control_flow_ctx
            if ctx and ctx.is_loop():
                return ctx
            layer = layer.parent
        return None

    @classmethod
    def inner_control_flow(cls) -> Optional[ControlFlowContext]:
        """
        :return: the most inner loop in the current context, if there is one
          E.g. you can use it to access the outer spatial dim.
        """
        return cls.top().control_flow_ctx()

    def __init__(
        self,
        *,
        module: Optional[rf.Module] = None,
        suggested_name: Optional[str] = None,
        name: Optional[str] = None,
        virtual: bool = False,
        can_access_children: bool = True,
        new_control_flow_ctx: Optional[ControlFlowContext] = None,
        parent: Optional[Layer] = NotSpecified,
    ):
        """
        You are not supposed to call this directly.
        Use :func:`NameCtx.new_root` or :func:`scoped`.
        """
        self.module = module
        self.tensor = None  # type: Optional[Tensor]
        self.tensor_remove_unused_cleanup_hooks = []  # type: List[Callable[[Tensor], None]]
        self.layer_dict = None  # type: Optional[LayerDictRaw]
        self.layer_extra_dependencies = []  # type: List[Layer]
        self.usages = []  # type: List[Layer]
        self.debug_layer = None  # type: Optional[LayerBase]
        self._enter_stack_frames = None  # type: Optional[Set[types.FrameType]]
        self.is_subnet = False  # it says whether it can have children
        self._subnet_main_output = None  # type: Optional[Tensor]  # when this is via SubnetworkLayer
        self.virtual = virtual  # does not consume a layer name in RETURNN. see get_name_in_ctx
        self.can_access_children = can_access_children  # from outside
        self.require_global_access = False  # from outside
        self.new_control_flow_ctx = new_control_flow_ctx
        self.children = {}  # type: Dict[str, Layer]
        self.extern_data = {}  # type: Dict[str, Tensor]  # only for the root name ctx
        self.global_batch = None  # type: Optional[BatchInfo]  # only for the root name ctx
        self.extra_net_dict = {}  # type: Dict[str, Any]  # only for the root name ctx
        self.marked_outputs = []  # type: List[Tensor[rfl.Layer]]
        self.marked_losses = []  # type: List[Tensor]
        self.parent = parent if parent is not NotSpecified else self.current_ctx()
        self.name = name  # early assign such that debug repr works later
        if not name:
            if suggested_name:
                suggested_name = suggested_name.replace("/", "_")
                suggested_name = tf_util.get_valid_scope_name_from_str(suggested_name)
                name = self._get_unique_name(suggested_name)
            elif self.parent:
                name = self._get_unique_name()
        self.name = name
        if self.parent:
            self.parent._add_child(self)
        self.custom_layer_name_scope = None  # type: Optional[str]  # layer_dict name_scope will be set to this
        self.__class__._recent = self

    def __repr__(self):
        parts = [self.get_abs_name_repr()]
        if self.tensor:
            parts.append("[%s]" % ",".join(self.tensor.get_batch_axes_short_description()))
        return f"<{self.__class__.__name__} {' '.join(parts)}>"

    def __hash__(self):
        return hash(id(self))

    def _sis_hash(self):
        from sisyphus.hash import sis_hash_helper  # noqa

        if not self.layer_dict:
            return sis_hash_helper(self.get_abs_name())
        return sis_hash_helper(self.layer_dict)

    def __copy__(self):
        """
        Normally we would not want to get a new name ctx with ``ctx != copy(ctx)``.

        :return: self
        :rtype: NameCtx
        """
        return self

    def __deepcopy__(self, memo=None):
        """
        Normally we would not want to get a new name ctx with ``ctx != deepcopy(ctx)``.

        :return: self
        :rtype: NameCtx
        """
        return self

    def assign_parent(self, parent: Layer, suggested_name: Optional[str] = None):
        """
        Assign or reassign parent to this name context.
        """
        if self.parent:
            self_ = self.parent.children.pop(self.name)
            assert self_ is self
            self.parent = None
        self.parent = parent
        self.name = self._get_unique_name(suggested_name or self.name)
        self.parent._add_child(self)

    def move_tensor_here(self: Layer, tensor: Tensor):
        """
        Moves an existing layer ref (with assigned name ctx)
        to another name ctx (without assigned layer or layer ref).

        This assumes that there are no other references to tensor.raw_tensor
        because those would become invalid.
        References to tensor itself should be fine.
        """
        assert not self.layer_dict and not self.tensor  # none yet assigned
        assert tensor.raw_tensor is not None
        assert isinstance(tensor.raw_tensor, Layer)

        # Remove tensor.name_ctx from its parent name ctx.
        if tensor.raw_tensor.parent:
            old_name_ctx = tensor.raw_tensor.parent.children.pop(tensor.raw_tensor.name)
            assert old_name_ctx is tensor.raw_tensor
        old_name_ctx: Layer = tensor.raw_tensor

        # Now reassign.
        self.tensor = tensor
        tensor.raw_tensor = self
        self.tensor_remove_unused_cleanup_hooks = old_name_ctx.tensor_remove_unused_cleanup_hooks
        self.layer_dict = old_name_ctx.layer_dict
        self.layer_extra_dependencies = old_name_ctx.layer_extra_dependencies
        self.usages = old_name_ctx.usages
        self.is_subnet = old_name_ctx.is_subnet
        self._subnet_main_output = old_name_ctx._subnet_main_output
        for name, child in old_name_ctx.children.items():
            child.parent = self
            if name not in self.children:
                self.children[name] = child
            else:
                name = child._get_unique_name(name)  # make sure name is unique
                child.name = name
                self.children[name] = child
        old_name_ctx.children = self.children  # just in case there is some other reference to the old name ctx

        if old_name_ctx.layer_dict:

            def _check_layer_opt_value(v):
                if isinstance(v, Net):
                    assert v.name_ctx is old_name_ctx
                    v.name_ctx = self

            nest.map_structure(_check_layer_opt_value, old_name_ctx.layer_dict)

    @property
    def root(self) -> Layer:
        """
        :return: root name ctx
        """
        root = self
        while root.parent:
            root = root.parent
        return root

    @property
    def is_root(self) -> bool:
        """
        :return: whether this is a root ctx
        """
        return not self.parent

    @property
    def can_access_children_from_root(self):
        """
        :return: whether can_access_children for self and all parents
        """
        name = self
        while name:
            if not name.can_access_children:
                return False
            name = name.parent
        return True

    def control_flow_ctx(self) -> Optional[ControlFlowContext]:
        """
        :return: control flow context of this name ctx
        """
        ctx = self
        while ctx:
            if ctx.new_control_flow_ctx:
                return ctx.new_control_flow_ctx
            ctx = ctx.parent
        return None

    def extend_reserved_names(self, names: Set[str]):
        """
        Extend reserved child names.
        """
        # Do not update inplace because we want an own instance on self.
        self._ReservedNames = self._ReservedNames | names

    def _assign_param_names(self, root_module: rf.Module):
        root = self.root
        for name, param in root_module.named_parameters(recurse=True):
            param = _resolve_param_tensor(param)
            param_layer = param.raw_tensor
            assert isinstance(param_layer, Layer), f"param {param} has no layer"
            if not param_layer.parent and param_layer is not root:  # no parent yet?
                param_layer.assign_parent(root, name)

    def _remove_unused_and_handle_subnets(self):
        # Collect all used tensor names.
        used_names = {self}  # type: Set[Layer]
        root = self.root
        queue = [
            (tensor, [])  # (tensor, path), where the path is how we get to the tensor through the graph, for debugging
            for tensor in self.marked_outputs + self.marked_losses
        ]  # type: List[Tuple[Tensor[Layer],List[Layer]]]
        while queue:
            tensor, src = queue.pop(0)
            if tensor.raw_tensor is None:
                raise Exception(f"tensor {tensor} has no layer defined, via {src}")
            # Parameters usually have no parent assigned at creation time.
            # However, we should have assigned them in _assign_param_names.
            if not tensor.raw_tensor.parent and tensor.raw_tensor != root:
                raise Exception(f"tensor {tensor} has no parent assigned, via {src}")
            if tensor.raw_tensor in used_names:
                continue
            used_names.add(tensor.raw_tensor)
            src_ = src + [tensor.raw_tensor]
            for dep in tensor.raw_tensor.get_tensor_dependencies():
                if dep.tensor is not None and dep not in used_names:
                    queue.append((dep.tensor, src_))

            # Handle subnetworks: Flatten away if just a single entry. Create layer if not created yet.
            ctx = tensor.raw_tensor  # type: Layer
            ctx.make_all_sub_networks_and_optimize()

            # Add tensor name including all parents.
            # Do this here after the potential late assign-parent and potential subnet flattening
            # because we need to know the right parents.
            for ctx in tensor.raw_tensor.get_abs_name_ctx_list():
                if ctx in used_names:
                    continue  # skip early, to skip the extra checks below
                if ctx.tensor is not None and ctx.tensor is not tensor:
                    queue.append((ctx.tensor, src_))

        # Go through all names in the hierarchy and remove unused.
        visited = set()  # type: Set[Layer]
        queue = [self]  # type: List[Layer]
        while queue:
            name_ctx = queue.pop(0)
            if name_ctx in visited:
                continue
            visited.add(name_ctx)
            if name_ctx not in used_names:
                assert name_ctx.parent
                name_ctx.parent.children.pop(name_ctx.name)
                if name_ctx.tensor is not None:
                    for hook in name_ctx.tensor_remove_unused_cleanup_hooks:
                        hook(name_ctx.tensor)
            else:
                for name, child in name_ctx.children.items():
                    assert child.parent is name_ctx and child.name == name
                    queue.append(child)

    def _prepare_for_config_serialization(self, *, root_module: rf.Module, name_path_cache: _NamePathCache):
        """
        Prepare the name ctx for RETURNN config serialization.
        This makes the root module maybe nicer and also removes unused entries.
        """
        assert self.root is self  # maybe not necessary but just assume this now
        # We want to flatten out root_module children into the root name space.
        # This is just for a nicer RETURNN net dict.
        # In the usual case, most of the net definition is in the root module call,
        # which is expected to be a subnetwork.
        # Doing this flattening is a bit tricky:
        # - The root namespace could still contain other stuff, and there might be name conflicts.
        #   This is still easy to resolve by making sure the names are unique.
        #   At this point, nothing should explicitly refer to any names.
        # - There might be references to the output of the root module call,
        #   e.g. for separately calculating the loss.
        #   So the root module call output must stay valid.
        #   Using self.root.move_tensor_here(...) would not really allow that the output is used
        #   because self.root does not have a valid layer name.
        root_module_calls = [child for child in self.root.children.values() if child.module is root_module]
        if root_module_calls:
            root_mod_call = root_module_calls[0]
            assert root_mod_call.module is root_module
            if root_mod_call is not self:
                # root_mod_call.layer might be None if the subnet is not yet initialized.
                if root_mod_call.tensor is not None:
                    assert not self.tensor  # not sure. maybe just reset?
                    assert root_mod_call.layer_dict["class"] == "subnetwork"
                    sub_out = root_mod_call.children.pop("output")
                    assert sub_out.layer_dict["class"] == "copy"
                    sub_real_out = sub_out.layer_dict["from"]
                    assert isinstance(sub_real_out, Tensor)
                    # Replace this tensor by the given tensor.
                    # This is a workaround in case other refs point to this tensor object.
                    sub_out.tensor = sub_real_out
                    root_mod_call.tensor = sub_real_out

                # Do not use self.move_tensor_here(root_mod_call.tensor) because we don't want the extra logic.
                for name, child in list(root_mod_call.children.items()):
                    child.assign_parent(parent=self, suggested_name=name)

        # Check if we can rename some layers
        # based on the module hierarchy (parameters and module outputs).
        # The layer names are not really relevant though,
        # so this is just to make the config nicer.
        # Only for parameters, we must make sure that the name is correct.
        # However, this can always be done via the `name_scope` option,
        # if the name would not match otherwise.
        queue = [self.root]  # type: List[Layer]
        visited: Set[Layer] = set()
        mod_in_layer = {}  # type: Dict[Tuple[Layer, RefIdEq[rf.Module]], Layer]
        while queue:
            ctx = queue.pop(0)
            if ctx in visited:  # can happen when parents are reassigned
                continue
            visited.add(ctx)

            assert not ctx.parent or ctx.parent.children[ctx.name] is ctx
            for child in ctx.children.values():
                queue.append(child)

            if not ctx.parent:
                continue

            if ctx.module is not None and not ctx.layer_dict:
                mod_path = name_path_cache.get_name_path(ctx.module, raise_exc=False)
                if mod_path is not None and len(mod_path) > 0 and mod_path[-1] != ctx.name:
                    existing = None  # type: Optional[Layer]
                    if (ctx.parent, RefIdEq(ctx.module)) in mod_in_layer:
                        existing = mod_in_layer[(ctx.parent, RefIdEq(ctx.module))]
                    elif mod_path[-1] in ctx.parent.children and ctx.parent.children[mod_path[-1]].module is ctx.module:
                        existing = ctx.parent.children[mod_path[-1]]
                    else:
                        for other in ctx.parent.children.values():
                            if other.module is ctx.module and other.name.startswith(mod_path[-1]):
                                existing = other
                                break
                    if existing:
                        assert existing.is_subnet and not existing.tensor and not existing.layer_dict
                        if ctx is existing:
                            pass  # we are already there
                        else:
                            if (ctx.parent, RefIdEq(ctx.module)) not in mod_in_layer:
                                mod_in_layer[(ctx.parent, RefIdEq(ctx.module))] = existing
                            # Move all ctx.children to existing.
                            for child in list(ctx.children.values()):
                                child.assign_parent(existing)
                            ctx.parent.children.pop(ctx.name)  # should not be needed anymore
                    else:
                        if mod_path[-1] not in ctx.parent.children:
                            # Rename ctx.name to mod_path[-1].
                            ctx.assign_parent(parent=ctx.parent, suggested_name=mod_path[-1])
                        elif ctx.name.startswith(mod_path[-1]):
                            pass  # just leave it
                        else:
                            # Rename ctx.name to suggested name based on mod_path[-1].
                            ctx.assign_parent(parent=ctx.parent, suggested_name=mod_path[-1])
                        mod_in_layer[(ctx.parent, RefIdEq(ctx.module))] = ctx

            elif ctx.tensor is not None:
                tensor_path = name_path_cache.get_name_path(ctx.tensor, raise_exc=False)
                if tensor_path is not None and len(tensor_path) > 0 and tensor_path[-1] != ctx.name:
                    if tensor_path[-1] not in ctx.parent.children:
                        # Rename ctx.name to tensor_path[-1].
                        ctx.assign_parent(parent=ctx.parent, suggested_name=tensor_path[-1])
                    elif ctx.name.startswith(tensor_path[-1]):
                        pass  # just leave it
                    else:
                        # Rename ctx.name to suggested name based on mod_path[-1].
                        ctx.assign_parent(parent=ctx.parent, suggested_name=tensor_path[-1])

        self._assign_param_names(root_module=root_module)
        self._remove_unused_and_handle_subnets()
        assert not self.parent, f"{self} get_returnn_config only makes sense in the root name ctx"

    def get_returnn_config(self, *, root_module: rf.Module) -> _ReturnnConfigSerializer:
        """
        :param root_module: there must be one root module such that all params have a well-defined name
        :return: config serializer
        """
        serializer = _ReturnnConfigSerializer(name_ctx=self, root_module=root_module)
        self._prepare_for_config_serialization(root_module=root_module, name_path_cache=serializer.name_path_cache)
        return serializer

    def make_net(self) -> Net:
        """
        Create new (sub) net, an instance of :class:`Net`.
        """
        return Net(name_ctx=self)

    def make_default_output(self, ref: Tensor) -> Tensor:
        """
        Assume this is a subnet, or the root net, and make a default output.
        """
        assert self.is_subnet
        if ref.raw_tensor is self.children.get(
            "output", None
        ):  # if this is the output layer already, allow and just return
            return ref
        assert "output" not in self.children
        return _utils.copy(ref, name=self.get_child("output"))

    def get_abs_name_ctx_list(self) -> List[Layer]:
        """
        Return list [root name ctx, ..., self].
        """
        ls = []
        cur = self
        while cur:
            ls.append(cur)
            cur = cur.parent
        return list(reversed(ls))

    def get_abs_name(self, *, join_str: str = "/") -> str:
        """
        :return: absolute RETURNN layer name starting from root context.
        """
        ls = self.get_abs_name_ctx_list()
        if len(ls) == 1:
            return ""
        assert len(ls) >= 2 and not ls[0].name and ls[-1] is self and ls[-1].name
        return join_str.join(ctx.name for ctx in ls[1:])

    def get_abs_name_repr(self) -> str:
        """
        :return: Some repr for our absolute name.
        """
        ls = self.get_abs_name_ctx_list()
        if len(ls) == 0:
            debug_name = "???"
        elif len(ls) == 1 and ls[0].name is None:
            debug_name = "/"
        else:
            debug_name = "/".join(
                (repr(ctx.name) if not ctx.virtual else f"({ctx.name!r})") if i > 0 or ctx.name is not None else ""
                for i, ctx in enumerate(ls)
            )
        return debug_name

    def get_name_in_ctx(self, ctx: Layer, *, middle_prefix: str = "", shorten_subnet: bool = True) -> str:
        """
        Get layer name valid in given scope.
        """
        assert not self.virtual and not self.is_root
        if self.parent is ctx:  # fast path
            return middle_prefix + self.name
        if self is ctx:
            return "base:" + self.get_name_in_ctx(
                ctx=ctx.parent, middle_prefix=middle_prefix, shorten_subnet=shorten_subnet
            )
        if isinstance(self.tensor, rfl.PrevTensorRef):
            return self.tensor.cur_layer_name_ctx.get_name_in_ctx(
                ctx, middle_prefix="prev:" + middle_prefix, shorten_subnet=False
            )
        ctx_scope_abs = ctx.get_abs_name_ctx_list()
        self_name_abs = self.get_abs_name_ctx_list()
        assert ctx_scope_abs[0] is self_name_abs[0]  # same root
        common_len = 0
        max_common_len = min(len(ctx_scope_abs), len(self_name_abs))
        while common_len < max_common_len and ctx_scope_abs[common_len] is self_name_abs[common_len]:
            common_len += 1
        del ctx_scope_abs[:common_len]
        del self_name_abs[:common_len]
        prefix = "".join(["base:" for ctx_ in reversed(ctx_scope_abs) if not ctx_.virtual])
        assert len(self_name_abs) >= 1, f"{self} in ctx {ctx} invalid"  # direct parent?
        assert self_name_abs[-1] is self
        if len(self_name_abs) == 1:  # fast path
            return prefix + middle_prefix + self.name
        if self.tensor is None or not shorten_subnet:  # without tensor, no further optimization
            postfix = "/".join([ctx.name for ctx in self_name_abs if not ctx.virtual])
            return prefix + middle_prefix + postfix
        # Potentially shorten postfix when it matches subnet outputs.
        while len(self_name_abs) >= 2:
            ctx_, ctx__ = self_name_abs[-2:]
            assert isinstance(ctx_, Layer) and isinstance(ctx__, Layer)
            if ctx_.layer_dict and ctx_.layer_dict["class"] == "subnetwork":
                if ctx_._subnet_main_output is ctx__.tensor or ctx_.children.get("output") is ctx__:
                    self_name_abs.pop(-1)
                    continue  # check again
            break
        postfix = "/".join([ctx_.name for ctx_ in self_name_abs if not ctx_.virtual])
        assert postfix, f"{self} in ctx {ctx} invalid, no postfix?"  # should not happen
        return prefix + middle_prefix + postfix

    def _add_child(self, child: Layer):
        assert child.name
        assert child.parent is self
        assert child.name not in self.children
        self.children[child.name] = child

    def get_child(self, name: str) -> Layer:
        """
        Makes sure the child exists.
        """
        if name in self.children:
            return self.children[name]
        else:
            return Layer(name=name, parent=self)  # also registers in self.children

    def get_new_child(self, suggested_name: str) -> Layer:
        """
        New child.
        """
        return Layer(suggested_name=suggested_name, parent=self)

    def get_child_with_tensor(self, name: str, *, data: Tensor) -> Layer:
        """
        Makes sure the child exists, including a corresponding layer ref.
        Creates the child together with a layer ref if it does not exist yet.
        """
        child = self.get_child(name)
        if not child.tensor:
            child.tensor = data
        assert child.tensor is data
        if data.raw_tensor is None:
            data.raw_tensor = child
        assert data.raw_tensor is child
        return child

    def get_child_tensor(self, name: str, *, data: Tensor) -> Tensor[Layer]:
        """
        Get child layer ref. Makes sure it exists.
        """
        return self.get_child_with_tensor(name, data=data).tensor

    def get_recent_tensor(self, *, only_same_control_flow: bool = False) -> Optional[Tensor]:
        """
        Get recent tensor if it exists. Can go deeply through children.
        """
        queue = [self]
        while queue:
            ctx = queue.pop(-1)  # depth-first
            if only_same_control_flow and ctx.control_flow_ctx() != self.control_flow_ctx():
                continue
            if ctx.tensor is not None:
                return ctx.tensor
            # due to pop(-1), this will be accessed in reverse order, which is what we want
            queue.extend(ctx.children.values())
        return None

    def __enter__(self):
        self._maybe_init_default_root()
        self._stack.append(self)
        from returnn.util.better_exchook import get_current_frame

        frame = get_current_frame()
        self._enter_stack_frames = set()
        while frame:
            self._enter_stack_frames.add(frame)
            frame = frame.f_back
        # make_layer() uses current_ctx() to get the parent scope.
        # current_ctx() uses _auto_setup_parent_name_ctx() and this checks the recent layer.
        # So, to make sure that we get the right parent scope in make_layer(),
        # we need to set this.
        self.__class__._recent = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._stack[-1] is self, f"{self}.__exit__: stack {self._stack} top is not self"
        self._enter_stack_frames = None
        self._stack.pop(-1)

    def _get_parent_module(self) -> Optional[rf.Module]:
        parent = self.parent
        while parent:
            if parent.module:
                return parent.module
            parent = parent.parent
        return None

    def _get_suggested_name(self) -> str:
        # https://github.com/rwth-i6/returnn_common/issues/125
        assert self.module is not None  # this function would not be used in another way
        parent_module = self._get_parent_module()
        if parent_module:
            # Check parent name scope module, any attrib from there to self.module.
            # Do a depth-first search through the parents, starting from self.module,
            # until we find self.parent.module.
            # Somewhat consistent to _get_abs_canonical_name.
            cache = _NamePathCache()
            cache.register_module(parent_module, [])
            path = cache.get_name_path(self.module, raise_exc=False)
            if path is not None:
                return ".".join(path)
        # Fallback to the canonical name.
        return self.module.get_default_name()

    def _get_unique_name(self, suggested_name: Optional[str] = None) -> str:
        name = suggested_name or self._get_suggested_name()
        reserved_names = set(self.parent.children.keys()) | self.parent._ReservedNames
        if self.parent.module:
            # Also reserve all attrib names of the parent module.
            # However, we allow to use the name if it is the attrib itself.
            if self.module and name not in reserved_names and getattr(self.parent.module, name, None) is self.module:
                return name
            if self.tensor and name not in reserved_names and getattr(self.parent.module, name, None) is self.tensor:
                return name
            # We might exclude all other attribs.
            # However, e.g. "dropout" is a common attrib storing the dropout rate (float),
            # and then when calling `rf.dropout`, it would not use that name, which is not what we want.
            # So, we only exclude attribs which do not have non-primitive types.
            for key, value in vars(self.parent.module).items():
                if not isinstance(value, (int, float, str, bool, type(None))):
                    reserved_names.add(key)
        if name not in reserved_names:
            return name
        i = 0
        while True:
            name_ = f"{name}_{i}"
            if name_ not in reserved_names:
                return name_
            i += 1

    def get_tensor_dependencies(self, *, _extra_layer_dict=None) -> List[Layer]:
        """
        :return: list of tensors this tensor depends on
        """
        dep_list = []
        dep_name_set = set()

        def _maybe_add_dep(x):
            if isinstance(x, Layer):
                if x in dep_name_set:
                    return
                dep_list.append(x)
                dep_name_set.add(x)
                return
            if isinstance(x, Tensor):
                return _maybe_add_dep(x.raw_tensor)
            if isinstance(x, Net):
                return _maybe_add_dep(x.name_ctx.children["output"].tensor)

        if _extra_layer_dict:
            nest.map_structure(_maybe_add_dep, _extra_layer_dict)
        if self.layer_dict:
            nest.map_structure(_maybe_add_dep, self.layer_dict)
        if self.children and "output" in self.children:
            _maybe_add_dep(self.children["output"].tensor)
        if self.parent and self.parent.tensor:
            _maybe_add_dep(self.parent.tensor)
        if self.layer_extra_dependencies:
            dep_list.extend(self.layer_extra_dependencies)
        return dep_list

    def make_all_sub_networks_and_optimize(self):
        """
        Go up all parents and create subnetworks which are not initialized yet.
        Also optimize by removing obsolete subnetworks (which just consist of one child).
        """
        ctx = self
        while True:
            if ctx.tensor is not None:
                ctx.optimize_move_up()
            ctx_ = ctx
            ctx = ctx_.parent
            if not ctx or ctx.is_root:
                break
            if ctx.virtual or ctx.tensor is not None or ctx_.tensor is None:
                continue
            if ctx.new_control_flow_ctx:
                continue
            ctx._make_sub_network_layer(ctx_.tensor)
            assert ctx.tensor is not None

    def optimize_move_up(self):
        """
        If the parent is a (non-initialized) subnet where we are the only child,
        move us up.
        """
        assert self.tensor is not None
        ctx = self.parent
        while ctx:
            assert isinstance(ctx, Layer)
            if not ctx._is_obsolete_subnet():
                break
            assert set(ctx.children.values()) == {self}
            ctx.parent.children[ctx.name] = self
            self.parent = ctx.parent
            self.name = ctx.name
            ctx = ctx.parent

    def _is_obsolete_subnet(self) -> bool:
        # Assume that we are not initialized yet (just for simplicity, not needed otherwise).
        if self.is_root or self.virtual or len(self.children) > 1:
            return False
        if self.tensor is not None:
            return False
        if self.module is not None and not isinstance(self.module, rf.Functional):
            return False
        return True

    def _make_sub_network_layer(self, sub_output: Tensor):
        assert self.tensor is None and self.layer_dict is None
        assert not self._is_obsolete_subnet()  # assume optimize_move_up() was called already
        if "output" in self.children:
            assert self.children["output"].tensor is sub_output
        else:
            if isinstance(sub_output, rfl.PrevTensorRef):
                # It would be quite confusing to have a prev-layer as default output,
                # so replace it by the current iteration.
                assert sub_output.cur_layer_name_ctx.tensor is not None
                sub_output = sub_output.cur_layer_name_ctx.tensor
            _utils.copy(sub_output, name=self.get_child("output"))
        rfl.make_layer(
            {"class": "subnetwork", "from": [], "subnetwork": self.make_net()},
            name=self,
            predefined_out_data=sub_output,
        )
        assert self.tensor is not None
        assert self.tensor.raw_tensor is self
        self._subnet_main_output = sub_output


class _ReturnnConfigSerializer:
    """
    Serializes a RETURNN config to a string.

    The config consists of generic RETURNN settings (behavior_version and maybe others)
    generic imports (e.g. "from returnn.tf.util.data import Data, Dim, ..."),
    dim tags, extern_data and the net dict.

    It is possible to first serialize only the part for extern_data (e.g. for the root config)
    including needed dim tags and imports,
    and separately serialize the net dict and remaining needed dim tags.
    """

    def __init__(self, *, name_ctx: Layer, root_module: rf.Module):
        """
        :param name_ctx:
        :param root_module: there must be one root module such that all params have a well-defined name
        """
        self.name_ctx = name_ctx
        self.root_module = root_module
        self.name_path_cache = _NamePathCache()
        self.name_path_cache.register_module(root_module, [])
        self._behavior_version = rfl.min_returnn_behavior_version
        self._dim_tags_proxy = ReturnnDimTagsProxy()
        self._base_extern_data_dim_refs = None  # type: Optional[List[ReturnnDimTagsProxy.DimRefProxy]]
        self._net_dict_builder = _NetDictBuilderCtx(root_module=self.root_module, name_path_cache=self.name_path_cache)

    def get_complete_py_code_str(self):
        """
        :return: complete combined config as Python code str.
          basically :func:`get_base_extern_data_py_code_str` + :func:`get_ext_net_dict_py_code_str`
        """
        return self.get_base_extern_data_py_code_str() + self.get_ext_net_dict_py_code_str(
            with_imports=False, ref_extern_data_dims_via_global_config=False
        )

    ImportPyCodeStr = (
        "from returnn.tf.util.data import (\n"
        "  Dim, batch_dim, single_step_dim,"
        " SpatialDim, FeatureDim, ImplicitDynSizeDim, ImplicitSparseDim)\n\n"
    )

    def get_base_extern_data_py_code_str(self) -> str:
        """
        :return: serialized config, i.e. Python code
        """
        assert self._base_extern_data_dim_refs is None  # only call once
        from returnn.util.pprint import pformat

        extern_data_raw = self.get_extern_data_raw_dict()
        extern_data_raw = self._dim_tags_proxy.collect_dim_tags_and_transform_config(extern_data_raw)
        self._base_extern_data_dim_refs = list(self._dim_tags_proxy.dim_refs_by_tag.values())

        code_lines = [
            self.ImportPyCodeStr,
            "use_tensorflow = True\n",
            f"behavior_version = {self._behavior_version}\n\n",
            f"{self._dim_tags_proxy.py_code_str()}\n",
            f"extern_data = {pformat(extern_data_raw)}\n",
        ]
        return "".join(code_lines)

    @classmethod
    def get_base_extern_data_py_code_str_direct(cls, extern_data: Dict[str, Any]) -> str:
        """
        directly get serialized Python code via extern data
        """
        dim_tags_proxy = ReturnnDimTagsProxy()
        from returnn.util.pprint import pformat

        extern_data = dim_tags_proxy.collect_dim_tags_and_transform_config(extern_data)

        code_lines = [
            cls.ImportPyCodeStr,
            f"{dim_tags_proxy.py_code_str()}\n",
            f"extern_data = {pformat(extern_data)}\n",
        ]
        return "".join(code_lines)

    def get_ext_net_dict_py_code_str(
        self, *, with_imports: bool = True, ref_extern_data_dims_via_global_config: bool = True
    ) -> str:
        """
        :param with_imports: whether to include imports
        :param ref_extern_data_dims_via_global_config: Add references to the definitions for the dimension tags
            written in `get_base_extern_data_py_code_str` via `returnn.config.get_global_config`.
        :return: serialized config, i.e. Python code
        """
        from returnn.util.pprint import pformat

        dim_tags_proxy = self._dim_tags_proxy.copy()
        net_dict = self.get_net_dict_raw_dict()
        net_dict = dim_tags_proxy.collect_dim_tags_and_transform_config(net_dict)
        imports = {}
        net_dict = self._post_process_transform(net_dict, imports=imports)
        code_lines = []

        if with_imports:
            code_lines.append(self.ImportPyCodeStr + "\n")
        for import_str in imports:
            code_lines.append(import_str + "\n")

        if ref_extern_data_dims_via_global_config:
            code_lines += ["from returnn.config import get_global_config\n", "config = get_global_config()\n"]
            for value in self._base_extern_data_dim_refs:
                code_lines.append(f"{value.py_id_name()} = config.typed_dict[{value.py_id_name()!r}]\n")

        code_lines += [
            f"{dim_tags_proxy.py_code_str(exclude_dims=self._base_extern_data_dim_refs)}\n",
            f"network = {pformat(net_dict)}\n",
        ]
        return "".join(code_lines)

    def get_net_dict_raw_dict(self) -> Dict[str, Any]:
        """
        :return: raw dict
        """
        return self._net_dict_builder.make_net_dict_raw(self.name_ctx.make_net())

    def get_extern_data_raw_dict(self) -> Dict[str, Any]:
        """
        :return: raw dict
        """
        return {
            data_key: {
                key: getattr(data, key)
                for key in [*data.get_kwargs(include_special_axes=False).keys(), "available_for_inference"]
                if key not in {"name", "batch"}
            }
            for (data_key, data) in self.name_ctx.extern_data.items()
        }

    def get_config_raw_dict(self) -> Dict[str, Any]:
        """
        :return: raw dict
        """
        return {
            "behavior_version": self._behavior_version,
            "extern_data": self.get_extern_data_raw_dict(),
            "network": self.get_net_dict_raw_dict(),
        }

    @classmethod
    def _post_process_transform(cls, obj, *, imports: Dict[str, None]):
        # imports is a dict to keep insertion order.
        # Similar as ReturnnDimTagsProxy.collect_dim_tags_and_transform_config.
        # Cannot use nest because nest does not support sets. Also nest requires them to be sorted.
        # See also NetDictBuilderCtx.make_net_dict_raw.
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        # We usually would be called after collect_dim_tags_and_transform_config, but we also allow it to be skipped.
        if isinstance(obj, (Dim, ReturnnDimTagsProxy.DimRefProxy, ReturnnDimTagsProxy.SetProxy)):
            return obj
        if isinstance(obj, numpy.ndarray):
            imports["import numpy"] = None
            return obj  # the standard repr of numpy arrays should work now
        import types

        if isinstance(obj, types.FunctionType):
            if obj.__module__.split(".")[0] != __name__.split(".")[0]:
                # Currently, we only allow functions from returnn_common to be used here,
                # as returnn_common is considered as stable,
                # and we do not serialize the function itself but just keep a ref to it here.
                # We can maybe later extend this whitelist to other packages such as TensorFlow.
                # For user code, we should serialize the function itself, which is not supported yet.
                raise ValueError(f"Function {obj} from unknown module {obj.__qualname__} cannot be serialized")
            imports[f"import {obj.__module__}"] = None
            return cls._CodeWrapper(f"{obj.__module__}.{obj.__qualname__}", obj)
        if isinstance(obj, dict):
            return {
                cls._post_process_transform(key, imports=imports): cls._post_process_transform(value, imports=imports)
                for key, value in obj.items()
            }
        if isinstance(obj, list):
            return [cls._post_process_transform(value, imports=imports) for value in obj]
        if isinstance(obj, tuple) and type(obj) is tuple:
            return tuple(cls._post_process_transform(value, imports=imports) for value in obj)
        if isinstance(obj, tuple) and type(obj) is not tuple:
            # noinspection PyProtectedMember,PyUnresolvedReferences,PyArgumentList
            return type(obj)(*(cls._post_process_transform(getattr(obj, key), imports=imports) for key in obj._fields))

    class _CodeWrapper:
        def __init__(self, code: str, obj: Any):
            self.code = code
            self.obj = obj

        def __repr__(self):
            return self.code


class _NetDictBuilderCtx:
    """
    Context for building the net.
    """

    def __init__(self, *, root_module: rf.Module, name_path_cache: _NamePathCache):
        self.root_module = root_module
        self.cache = name_path_cache

    class _StackInfo:
        def __init__(
            self,
            *,
            parent: Optional[_NetDictBuilderCtx._StackInfo] = None,
            net: Net,
            layer_abs_name_scope_effective: str,
        ):
            self.parent = parent
            self.net = net
            self.layer_abs_name_scope_effective = layer_abs_name_scope_effective

        def add(self, *, net: Net, layer_abs_name_scope_effective: str) -> _NetDictBuilderCtx._StackInfo:
            """
            :return: new stack info
            """
            return _NetDictBuilderCtx._StackInfo(
                parent=self, net=net, layer_abs_name_scope_effective=layer_abs_name_scope_effective
            )

        def get_parent_loop_axes(self) -> List[Dim]:
            """
            via control flow ctx
            """
            dims = []
            parent = self
            while parent:
                ctx = parent.net.name_ctx.control_flow_ctx()
                if ctx:
                    if ctx.is_loop():
                        if ctx.loop_spatial_dim is not None and ctx.loop_spatial_dim not in dims:
                            dims.append(ctx.loop_spatial_dim)
                parent = parent.parent
            return list(reversed(dims))

    def make_net_dict_raw(self, net: Net, *, _stack: Optional[_StackInfo] = None) -> NetDictRaw:
        """
        Create raw net dict, not containing any :class:`Tensor` or :class:`Net` instances anymore.
        """
        import types

        if _stack is None:
            _stack = self._StackInfo(net=net, layer_abs_name_scope_effective="")
        net_dict = {}
        for sub_name_ctx in net.name_ctx.children.values():
            if not sub_name_ctx.layer_dict:
                continue

            layer_dict = sub_name_ctx.layer_dict.copy()
            assert "class" in layer_dict

            data_template = sub_name_ctx.tensor.copy_template()
            for outer_dim in _stack.get_parent_loop_axes():
                if outer_dim in data_template.dim_tags:
                    data_template = data_template.copy_template_excluding_axis(
                        data_template.get_axis_from_description(outer_dim)
                    )
            dim_tags = list(data_template.dim_tags)
            for dim in dim_tags:
                if dim.is_batch_dim() or dim.is_static():
                    continue
                # We need dyn_size_ext to know the implicit dims, to correctly set out_shape.
                # If dyn_size_ext is not set yet, try to complete it.
                if not dim.dyn_size_ext:
                    dim.complete_dyn_size()
                assert (
                    dim.dyn_size_ext
                ), f"{sub_name_ctx}: need {dim} to be defined to be able to know about implicit dims"
            dim_tags.extend(data_template.dim_tags_set_implicit_only_wrapped)
            assert len(dim_tags) == len(
                set((d, d.match_priority if isinstance(d, Dim) else 0) for d in dim_tags)
            ), f"duplicate dims in {sub_name_ctx} {sub_name_ctx.tensor}"
            if len(dim_tags) == len(set(dim_tags)):  # might not be unique without match_priority
                # For some layer classes, the out_shape would be redundant.
                if layer_dict["class"] not in {"constant", "variable", "random", "subnetwork", "transpose"}:
                    layer_dict["out_shape"] = set(dim_tags)

            assert "name_scope" not in layer_dict  # we explicitly want to assign it now (if needed)
            if sub_name_ctx.custom_layer_name_scope is not None:
                sub_name_scope = sub_name_ctx.custom_layer_name_scope
                layer_dict["name_scope"] = sub_name_scope
                assert sub_name_scope == ""  # anything else unexpected currently
                sub_layer_abs_name_scope = _stack.layer_abs_name_scope_effective
            else:
                # We must check whether the RETURNN abs layer name is consistent with our module naming hierarchy,
                # and make it consistent if not (https://github.com/rwth-i6/returnn_common/issues/25).
                # The parent name ctx RETURNN layer will also have the right name_scope set,
                # so this layers name scope default is simply based on that.
                # Note that parameters could be assigned lazily at some later point.
                layer_abs_name_scope_parent = _stack.layer_abs_name_scope_effective
                if layer_abs_name_scope_parent:
                    layer_abs_name_scope_parent += "/"
                layer_abs_name_scope_default = layer_abs_name_scope_parent + sub_name_ctx.name

                sub_layer_abs_name_scope = self._expected_layer_abs_name_scope(sub_name_ctx)
                if sub_name_ctx.layer_dict["class"] == "variable":
                    assert (
                        sub_layer_abs_name_scope
                    ), f"VariableLayer {sub_name_ctx} must have a unique name in {self.root_module}"
                if sub_layer_abs_name_scope is not None:
                    if (
                        layer_abs_name_scope_default != sub_layer_abs_name_scope
                    ):  # default does not match what we require
                        if sub_layer_abs_name_scope == _stack.layer_abs_name_scope_effective:
                            layer_dict["name_scope"] = ""
                        elif sub_layer_abs_name_scope.startswith(layer_abs_name_scope_parent):  # can use relative
                            layer_dict["name_scope"] = sub_layer_abs_name_scope[len(layer_abs_name_scope_parent) :]
                        else:  # must use absolute
                            layer_dict["name_scope"] = "/" + sub_layer_abs_name_scope
                else:
                    sub_layer_abs_name_scope = layer_abs_name_scope_default

            def _map_elem_resolve(obj: Any) -> Any:
                if isinstance(obj, Tensor):
                    assert isinstance(
                        obj.raw_tensor, rfl.Layer
                    ), f"unexpected tensor {obj} with raw tensor type {type(obj.raw_tensor)}, expected rfl.Layer"
                    obj: Tensor[rfl.Layer]
                    assert obj.raw_tensor.parent or net.name_ctx == obj.raw_tensor
                    return obj.raw_tensor.get_name_in_ctx(ctx=net.name_ctx)
                if isinstance(obj, Net):
                    return self.make_net_dict_raw(
                        net=obj, _stack=_stack.add(net=obj, layer_abs_name_scope_effective=sub_layer_abs_name_scope)
                    )
                # We assume only basic types. This is not really a restriction but just a sanity check.
                # You might want to extend this.
                # However, then make sure that serialization to string is handled in ReturnnConfigSerializer.
                assert isinstance(
                    obj, (int, float, str, bool, numpy.ndarray, set, Dim, type(None), types.FunctionType)
                ), f"unexpected type {type(obj)}"
                if isinstance(obj, Dim) and obj.is_batch_dim():
                    return batch_dim
                return obj

            layer_dict = nest.map_structure(_map_elem_resolve, layer_dict)
            net_dict[sub_name_ctx.name] = layer_dict
        net_dict.update(net.name_ctx.extra_net_dict)
        return net_dict

    def _expected_layer_abs_name_scope(self, name_ctx: Layer) -> Optional[str]:
        """
        :param NameCtx name_ctx:
        :return: expected absolute name scope for this layer
        """
        if name_ctx.custom_layer_name_scope is not None:
            if name_ctx.custom_layer_name_scope == "":
                if name_ctx.parent:
                    return self._expected_layer_abs_name_scope(name_ctx.parent)
                else:
                    return ""
            raise NotImplementedError(f"custom_layer_name_scope {name_ctx.custom_layer_name_scope!r} not supported yet")

        if name_ctx.tensor is not None:
            name_path_tensor = self.cache.get_name_path(name_ctx.tensor, raise_exc=False)
            if name_path_tensor is not None:
                return "/".join(name_path_tensor)
        if name_ctx.module:
            name_path_mod = self.cache.get_name_path(name_ctx.module, raise_exc=False)
            if name_path_mod is not None:
                return "/".join(name_path_mod)

        return None


class Net:
    """
    Represents a RETURNN (sub) network.
    """

    def __init__(self, *, name_ctx: Layer):
        self.name_ctx = name_ctx

    def __repr__(self):
        return f"Net{self.name_ctx!r}"


class ReturnnDimTagsProxy:
    """
    When serialized via __repr__, this represents a dict unique_name -> dim tag.
    All usages in the network and extern_data will also get proxies when serialized point to this dict.
    """

    class DimRefProxy:
        """
        This will be a reference to the global dim_tags __repr__.
        """

        def __init__(
            self,
            *,
            dim: Union[Dim, _MarkedDim],
            name: Optional[str],
            path: Tuple[Any, ...],
            parent: ReturnnDimTagsProxy,
        ):
            self._dim = dim
            self.name = name  # None, or valid Python identifier
            self.path = path
            self.parent = parent
            self.debug_idx = len(parent.dim_refs_by_name)

        def __repr__(self):
            return self.ref_repr()

        def _sis_hash(self):
            # noinspection PyUnresolvedReferences,PyPackageRequirements
            from sisyphus.hash import sis_hash_helper

            return sis_hash_helper(self.path)

        @property
        def dim(self) -> Dim:
            """Dim"""
            if isinstance(self._dim, Dim):
                return self._dim
            elif isinstance(self._dim, _MarkedDim):
                return self._dim.tag
            else:
                raise TypeError(f"invalid {self._dim}")

        def ref_repr(self) -> str:
            """ref repr"""
            return self.parent.dim_ref_repr(self._dim, brackets=False, prefer_ref=True)

        def py_id_name(self) -> str:
            """
            :return: valid Python identifier
            """
            assert self.name
            return self.name + "_dim"

        def dim_repr(self):
            """
            Dim repr, used for serialization of all registered dim tags.
            Any derived dims or special dims will not be registered and instead be represented
            with the same derivation referencing other registered dim tags.
            See :func:`ReturnnDimTagsProxy.dim_ref_repr`.
            """
            dim = self._dim
            if isinstance(dim, _MarkedDim):
                return self.parent.dim_ref_repr(dim, brackets=False, prefer_ref=False)
            assert isinstance(dim, Dim)
            assert not dim.is_batch_dim()
            assert dim.can_be_used_as_dim()
            if dim.derived_from_op:
                return self.parent.dim_ref_repr(dim, brackets=False, prefer_ref=False)
            assert not dim.match_priority
            # We assume FeatureDim, SpatialDim and Dim are imported.
            if dim.kind == Dim.Types.Feature:
                return f"FeatureDim({dim.description!r}, {dim.dimension})"
            if dim.kind == Dim.Types.Spatial:
                if dim.dimension is not None:
                    return f"SpatialDim({dim.description!r}, {dim.dimension})"
                else:
                    return f"SpatialDim({dim.description!r})"
            # generic fallback
            return f"Dim(kind={dim.kind}, description={dim.description!r}, dimension={dim.dimension})"

    class SetProxy:
        """
        This represents a set but with a predefined order.
        We want a deterministic order in the repr such that the generated code stays deterministic.
        """

        def __init__(self, values: Sequence[Any]):
            self.values = values

        def __repr__(self):
            return f"{{{', '.join(map(repr, self.values))}}}"

    # --------- ReturnnDimTagsProxy ---------------

    def __init__(self, *, reserved_names: Optional[Set[str]] = None):
        self.dim_refs_by_name = {}  # type: Dict[str, ReturnnDimTagsProxy.DimRefProxy]
        self.dim_refs_by_tag = {}  # type: Dict[Dim, ReturnnDimTagsProxy.DimRefProxy]
        # You can externally set this to some other set, or add names to it.
        # Note that we will also add names to this instance.
        self.reserved_names = reserved_names or set()  # type: Set[str]
        self.reserved_names.update(
            {
                "batch_dim",
                "single_step_dim",
                "Data",
                "Dim",
                "FeatureDim",
                "SpatialDim",
                "ImplicitSparseDim",
                "ImplicitDynSizeDim",
            }
        )

    def __repr__(self):
        return "\n".join(
            [
                f"<{self.__class__.__name__}:",
                *(f"  {value.py_id_name()} = {value.dim_repr()}" for key, value in self.dim_refs_by_name.items()),
                ">",
            ]
        )

    def copy(self) -> ReturnnDimTagsProxy:
        """
        :return: creates a shallow copy
        """
        new = ReturnnDimTagsProxy()
        new.dim_refs_by_name = self.dim_refs_by_name.copy()
        new.dim_refs_by_tag = self.dim_refs_by_tag.copy()
        new.reserved_names = self.reserved_names.copy()
        return new

    def py_code_str(self, exclude_dims: Collection[ReturnnDimTagsProxy.DimRefProxy] = ()):
        """
        :param exclude_dims: dim tags to exclude from serializing
        :return: Python code
        """
        # We cannot just iterate through self.dim_refs_by_name in insertion order
        # because the derived_from_op references tags might only be referenced later.
        visited = set()  # type: Set[str]  # names of already visited tags
        lines = []

        def _visit_tag_deps(tag: Dim):
            if tag.derived_from_op:
                for tag_ in tag.derived_from_op.inputs:
                    if tag_ in self.dim_refs_by_tag:
                        _visit_ref(self.dim_refs_by_tag[tag_])  # make sure to visit it first
                    else:
                        _visit_tag_deps(tag_)

        def _visit_ref(ref: ReturnnDimTagsProxy.DimRefProxy):
            if ref in exclude_dims:
                return
            _visit_tag_deps(ref.dim)
            if ref.name in visited:
                return
            visited.add(ref.name)
            lines.append(f"{ref.py_id_name()} = {ref.dim_repr()}\n")

        for _, value in self.dim_refs_by_name.items():
            _visit_ref(value)

        return "".join(lines)

    def _sis_hash(self):
        raise Exception("unexpected")

    def dim_ref_repr(self, dim: Union[Dim, _MarkedDim], *, brackets: bool = True, prefer_ref: bool = True) -> str:
        """
        :return: for the given dim, Python code which refers to it, via ``dim_tags``
        """
        if isinstance(dim, _MarkedDim):
            return f"{dim.__class__.__name__}({self.dim_ref_repr(dim.tag, brackets=False, prefer_ref=prefer_ref)})"
        assert isinstance(dim, Dim)
        if dim == batch_dim:
            return "batch_dim"
        if dim == single_step_dim:
            return "single_step_dim"
        if dim.match_priority:
            return f"{self.dim_ref_repr(dim.copy(match_priority=0))}.copy(match_priority={dim.match_priority})"
        if not dim.derived_from_op and dim.get_same_base().derived_from_op:
            dim = dim.get_same_base()
        ref = self.dim_refs_by_tag.get(dim)
        if prefer_ref and ref:
            return ref.py_id_name()
        if dim.derived_from_op:
            if dim.derived_from_op.kind == "constant":
                v = dim.derived_from_op.attribs["value"]
                if v < 0 and brackets:
                    return f"({v})"
                return str(v)
            func_map = {"truediv_left": "div_left", "ceildiv_left": "ceildiv_left", "ceildiv_right": "ceildiv_right"}
            if dim.derived_from_op.kind in func_map:
                assert len(dim.derived_from_op.inputs) == 2
                a, b = dim.derived_from_op.inputs
                return f"{self.dim_ref_repr(a)}.{func_map[dim.derived_from_op.kind]}({self.dim_ref_repr(b)})"
            op_str = {"add": "+", "mul": "*", "truediv_right": "//", "floordiv_right": "//"}[dim.derived_from_op.kind]
            s = f" {op_str} ".join(self.dim_ref_repr(in_) for in_ in dim.derived_from_op.inputs)
            return f"({s})" if brackets else s
        assert ref, f"no ref for {dim}"
        return ref.py_id_name()

    def collect_dim_tags_and_transform_config(self, config: T) -> T:
        """
        Go through the config and collect all dim tags, replace them by proxies (DimRefProxy or SetProxy).

        :return: new config
        """
        import re

        def _sort_key(value):
            if isinstance(value, ReturnnDimTagsProxy.DimRefProxy):
                if value.dim.kind == Dim.Types.Batch:
                    return -1
                return value.debug_idx
            return value

        def _unique_name(dim: Dim) -> str:
            assert dim not in self.dim_refs_by_tag
            name_ = dim.description
            name_ = re.sub(r"[^a-zA-Z0-9_]", "_", name_)
            if name_.endswith("_dim"):
                name_ = name_[: -len("_dim")]
            if not name_ or name_[:1].isdigit():
                name_ = "_" + name_
            if name_ not in self.reserved_names:
                return name_
            i = 0
            while True:
                name__ = f"{name_}_{i}"
                if name__ not in self.reserved_names:
                    return name__
                i += 1

        # Cannot use nest because nest does not support sets. Also nest requires them to be sorted.
        def _map(path, value, *, direct=True):
            if isinstance(value, _MarkedDim):
                _map(path, value.tag)  # Register the dim tag
                return ReturnnDimTagsProxy.DimRefProxy(dim=value, name=None, path=path, parent=self)
            if isinstance(value, Dim):
                if value in {batch_dim, single_step_dim}:
                    # No need to register this.
                    return ReturnnDimTagsProxy.DimRefProxy(dim=value, name=None, path=path, parent=self)
                if value.match_priority != 0:
                    _map(path, value.copy(match_priority=0))  # Register the dim tag without match_priority.
                    # Now return the custom proxy for the dim tag with match_priority. No need to register this.
                    return ReturnnDimTagsProxy.DimRefProxy(dim=value, name=None, path=path, parent=self)
                value = value.get_same_base()
                if value.derived_from_op:
                    # Make sure all the inputs are registered.
                    for i, child in enumerate(value.derived_from_op.inputs):
                        _map(path + (value.derived_from_op.kind, i), child, direct=False)
                    # No need to register this.
                    if not direct:
                        return ReturnnDimTagsProxy.DimRefProxy(dim=value, name=None, path=path, parent=self)
                    # However, pass on to register this anyway.
                    # While this would not be explicitly needed, as we can directly refer to it,
                    # this is still nicer to see all dim tags explicitly.
                if value in self.dim_refs_by_tag:
                    return self.dim_refs_by_tag[value]
                name = _unique_name(value)
                assert name not in self.dim_refs_by_name
                ref = ReturnnDimTagsProxy.DimRefProxy(dim=value, name=name, path=path, parent=self)
                self.dim_refs_by_name[name] = ref
                self.dim_refs_by_tag[value] = ref
                self.reserved_names.add(name)
                return ref
            if isinstance(value, dict):
                return {
                    _map(path + (key, "key"), key): _map(path + (key, "value"), value_) for key, value_ in value.items()
                }
            if isinstance(value, list):
                return [_map(path + (i,), value_) for i, value_ in enumerate(value)]
            if isinstance(value, tuple) and type(value) is tuple:
                return tuple(_map(path + (i,), value_) for i, value_ in enumerate(value))
            if isinstance(value, tuple) and type(value) is not tuple:
                # noinspection PyProtectedMember,PyUnresolvedReferences,PyArgumentList
                return type(value)(*(_map(path + (key,), getattr(value, key)) for key in value._fields))
            if isinstance(value, set):
                values = [_map(path + (value,), value_) for value_ in value]
                values.sort(key=_sort_key)  # this should be possible now because it would be some sortable proxies
                return ReturnnDimTagsProxy.SetProxy(values)
            return value

        config = _map((), config)
        return config


class _NamePathCache:
    def __init__(self):
        self.module_to_name_path = {}  # type: Dict[RefIdEq[rf.Module], Tuple[str, ...]]  # module -> full name path
        self.tensor_to_name_path = {}  # type: Dict[rfl.Layer, Tuple[str, ...]]  # tensor (layer) -> full name path
        # (Tensor is not hashable, thus use its Layer)
        self.name_path_to_module = {}  # type: Dict[Tuple[str, ...], rf.Module]  # full name path -> module

    def register_module(self, module: rf.Module, name_path: Sequence[str]):
        """
        Register some module (e.g. root module).
        """
        assert isinstance(module, rf.Module)
        assert isinstance(name_path, (tuple, list))
        assert RefIdEq(module) not in self.module_to_name_path
        self.module_to_name_path[RefIdEq(module)] = tuple(name_path)
        self.name_path_to_module[tuple(name_path)] = module

        queue = [module]
        while queue:
            parent = queue.pop(0)
            for name, child in parent.named_children():
                if RefIdEq(child) not in self.module_to_name_path:
                    self.module_to_name_path[RefIdEq(child)] = self.module_to_name_path[RefIdEq(parent)] + (name,)
                    self.name_path_to_module[self.module_to_name_path[RefIdEq(child)]] = child
                    queue.append(child)
            for name, param in parent.named_parameters(recurse=False):
                assert isinstance(param.raw_tensor, rfl.Layer)
                param = _resolve_param_tensor(param)
                if param.raw_tensor not in self.tensor_to_name_path:
                    self.tensor_to_name_path[param.raw_tensor] = self.module_to_name_path[RefIdEq(parent)] + (name,)

    def get_name_path(
        self: _NamePathCache,
        child: Union[rf.Module, Tensor],
        *,
        raise_exc: bool = True,
    ) -> Optional[Tuple[str, ...]]:
        """
        :return: unique absolute layer name for the module hierarchy.
          https://github.com/rwth-i6/returnn_common/issues/25
          https://github.com/rwth-i6/returnn_common/issues/125
        """
        assert self.module_to_name_path  # call register_module first
        if isinstance(child, Tensor):
            if raise_exc:
                return self.tensor_to_name_path[child.raw_tensor]
            else:
                return self.tensor_to_name_path.get(child.raw_tensor)
        elif isinstance(child, rf.Module):
            if raise_exc:
                return self.module_to_name_path[RefIdEq(child)]
            else:
                return self.module_to_name_path.get(RefIdEq(child))
        else:
            raise TypeError(f"invalid type {type(child)}")


def _resolve_param_tensor(param: rf.Parameter[rfl.Layer]) -> rf.Tensor[rfl.Layer]:
    """
    Get the original tensor from a parameter, pointing to the VariableLayer.
    Via parameter_assign, the current param tensor might be some variable read,
    not the original VariableLayer.

    :param param:
    :return: tensor pointing to the VariableLayr
    """
    while True:
        if param.raw_tensor.layer_dict["class"] == "variable":
            return param
        if param.raw_tensor.layer_dict["class"] == "variable_read":
            param = param.raw_tensor.layer_dict["var"]
            assert isinstance(param, rf.Tensor)
            continue
        raise Exception(f"unexpected param tensor {param} {param.raw_tensor} with opts {param.raw_tensor.layer_dict}")


_AutoSetupNameCtxPrevTopFrame = None  # type: Optional[types.FrameType]
_AutoSetupNameCtxCodeBlacklist = set()  # type: Set[types.CodeType]


def _auto_setup_parent_name_ctx(*, ignore_top_stack_frames: int = 1) -> Layer:
    """
    Sets up a NameCtx corresponding to the Python call stack trace.

    From the call stack, we consider methods from modules (rf.Module subclasses)
    or global functions on tensors.

    There are some heuristics involved but this should not be critical.

    https://github.com/rwth-i6/returnn_common/issues/159

    :param ignore_top_stack_frames:
    :return: name ctx for the layer
    """
    global _AutoSetupNameCtxPrevTopFrame
    from returnn.util.better_exchook import get_current_frame

    frame = get_current_frame()
    assert frame
    code_blacklist = {
        _auto_setup_parent_name_ctx.__code__,
        Layer.__init__.__code__,
        Layer.current_ctx.__code__,
        rfl.make_layer.__code__,
    }
    code_blacklist.update(_AutoSetupNameCtxCodeBlacklist)
    ignore_top_stack_frames += 1  # ignore ourself
    while ignore_top_stack_frames > 0:
        assert frame.f_back
        frame = frame.f_back
        ignore_top_stack_frames -= 1
    while frame.f_code in code_blacklist:
        assert frame.f_back
        frame = frame.f_back
    top_frame = frame  # e.g. the caller of make_layer

    prev_frames = set()
    frame = _AutoSetupNameCtxPrevTopFrame
    while frame:
        prev_frames.add(frame)
        frame = frame.f_back

    cur_ctx = Layer.recent_subnet()
    if not cur_ctx.is_subnet:
        assert cur_ctx.parent and cur_ctx.parent.is_subnet
        cur_ctx = cur_ctx.parent
    assert cur_ctx.is_subnet
    cur_control_flow_ctx = cur_ctx.control_flow_ctx()
    cur_root_ctx = cur_ctx.root

    ctx = None  # type: Optional[Layer]
    module_ids = set()  # avoid duplicates
    module_frames = []  # type: List[rf.Module]
    frame = top_frame
    while frame:
        # Stop in the frame from the cur context.
        # Ignore the cur context if it is the root because the root creation stack trace can be arbitrary.
        # noinspection PyProtectedMember
        if cur_ctx.parent and cur_ctx._enter_stack_frames and frame in cur_ctx._enter_stack_frames:
            break
        if frame.f_code in code_blacklist:
            frame = frame.f_back
            continue

        # find module from module method or function
        mod = None
        # In case of a method, func will point to the original function (FunctionType), not the MethodType.
        # We use a more generic method: The first argument of the function (frame.f_code.co_varnames[0])
        # is usually `self` in a method. If this is a module, we use it.
        if frame.f_code.co_varnames and isinstance(frame.f_locals.get(frame.f_code.co_varnames[0]), rf.Module):
            mod = frame.f_locals[frame.f_code.co_varnames[0]]
        if mod is not None and id(mod) not in module_ids:
            calls = [
                layer
                for layer in cur_ctx.get_abs_name_ctx_list() + list(cur_ctx.children.values())
                if layer.module is mod
            ]
            if calls:
                # We can reuse some existing name ctx.
                ctx = calls[0]
                break
            module_frames.append(mod)
            module_ids.add(id(mod))

        frame = frame.f_back

    if ctx is None:
        ctx = cur_ctx if cur_control_flow_ctx else cur_root_ctx

    for module in reversed(module_frames):
        # Note: instead of just storing the module, we could also cleverly infer a good suggested name
        #   by looking at the code and check for patterns like "whatever = func(...)"
        #   and then use "whatever" as suggested name.
        ctx = Layer(module=module, parent=ctx)
        ctx.is_subnet = True

    _AutoSetupNameCtxPrevTopFrame = top_frame
    return ctx


def auto_setup_name_ctx_ignore_func(func: Union[types.FunctionType, Callable]):
    """
    Registers the func in the blacklist.
    """
    _AutoSetupNameCtxCodeBlacklist.add(func.__code__)
