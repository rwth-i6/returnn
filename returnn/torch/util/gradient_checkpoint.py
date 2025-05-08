"""
Gradient checkpointing.

Following a lot of the code of the official
`torch.utils.checkpoint <https://pytorch.org/docs/stable/checkpoint.html>`__,
using ``torch.autograd.graph.saved_tensors_hooks``
and ``TorchDispatchMode``
but also handling the RNG fork and reset in a similar way.

See also :mod:`returnn.tf.util.gradient_checkpoint`:
same API and logic in TF, although it heavily makes use
of the TF computation graph, i.e. graph mode,
which makes this particular feature much easier to implement.

See also:
https://github.com/rwth-i6/returnn/issues/1552
https://discuss.pytorch.org/t/gradient-checkpointing/205416
https://gist.github.com/soulitzer/ec1049a947be046de7fbc2af61a4ee8c
"""

from __future__ import annotations

from typing import Optional, Union, Any, Callable, Sequence, List, Dict
from types import MethodType
from dataclasses import dataclass, field
import contextlib
from weakref import ref, WeakSet
import threading
import atexit

import torch
from torch.utils.weak import WeakTensorKeyDictionary  # needs Torch >=2.0.0

# noinspection PyProtectedMember
from torch.utils._python_dispatch import TorchDispatchMode

# PyTree is very common and semi-standard for PyTorch, e.g. __torch_dispatch__.
# We might use dm-tree or so alternatively here, but PyTree should be fine.
# noinspection PyProtectedMember
import torch.utils._pytree as pytree


__all__ = ["gradient_checkpoint_scope"]


# gradient_checkpoint_scope is the public API to the user.
# gradient_checkpoint_scope.__enter__ will enter two other scopes:
#
# - record_graph_scope: _RecordGraph(TorchDispatchMode),
#   to record the computation graph for all ops within the scope.
#
# - saved_tensors_hooks_scope: torch.autograd.graph.saved_tensors_hooks,
#   to overwrite what we store for backpropagation, and how to recompute it.
#   Specifically, for all tensors which were created within the gradient_checkpoint_scope,
#   we will never store them in the pack_hook,
#   and unpack_hook will trigger the recomputation of the computation graph.
#
# gradient_checkpoint_scope.__exit__ will exit the record_graph_scope,
# but the saved_tensors_hooks_scope will stay alive as long as needed,
# while any of the created tensors are still alive.
# We keep a weak tensor key dictionary to map from the created raw tensors
# to the point in the recorded computation graph (specifically _GraphTensor objects).
# We just check whether any of the weak tensor refs is still alive.
#
# To keep saved_tensors_hooks_scope alive and make sure
# that other calls to torch.autograd.graph.saved_tensors_hooks are correctly handled,
# specifically that the order of enter/exit is correct,
# we hook into torch.autograd.graph.saved_tensors_hooks.__enter__/__exit__ itself.
# See _register_custom_saved_tensors_hooks below.
# Further, torch.autograd.graph.saved_tensors_hooks is thread local,
# so we can do any such logic only within the same thread.
# We also hook into Tensor.__del__ and also handle gradient_checkpoint_scope.__del__,
# but as that might run in a different thread, we cannot always do the cleanup there.
# We always check for this.
# (Note that this is due to the API of torch.autograd.graph.saved_tensors_hooks.
# We actually would want to always use it for a set of specified tensors.
# We also discuss some potentially better PyTorch API to implement this in an easier way:
# https://github.com/pytorch/pytorch/issues/129867)
#
# For the recomputation, we make sure that we properly reset the RNG and AMP states,
# and that we perform the recomputation in the exact same order, such that RNG state is correct.
#
# Once some recomputed tensor was used and is not needed anymore, the GC should free it.
# We try to make sure that no unnecessary references are kept alive.
#
# Also see test_gradient_checkpoint_scope() which tests this.


class gradient_checkpoint_scope:
    """
    Create a gradient checkpoint scope.
    All tensors created within this scope will not be stored for backpropagation,
    but will be recomputed on the fly during backpropagation.

    Example::

        a = ...
        b = ...
        c = ...
        with gradient_checkpoint_scope():
            x = a + b
        y = x * c

    In this example, the tensor ``x`` will not be stored for backpropagation,
    i.e. the computation ``x = a + b`` will be recomputed during backpropagation.

    Internally, this uses the PyTorch ``torch.autograd.graph.saved_tensors_hooks`` mechanism
    to override what we store for backpropagation, and how to recompute it.
    And we use the PyTorch ``TorchDispatchMode`` to intercept all operations within the scope.
    Note that the usage of ``torch.autograd.graph.saved_tensors_hooks`` is tricky here
    as we need it beyond the scope of the ``gradient_checkpoint_scope``,
    specifically for all future usages of the tensor ``x`` in the example.
    See the code documentation for more details on this.

    Note, PyTorch itself also provides a gradient checkpointing API,
    namely `torch.utils.checkpoint <https://pytorch.org/docs/stable/checkpoint.html>`__.
    This API is different: You cannot easily specify what not to store / what to recompute.
    You rather specify a start/end point what to *store* for backpropagation,
    and then PyTorch will recompute everything in between.
    For the example above, you define that ``y`` is the end point and will be stored.
    It looks like this::

        a = ...
        b = ...
        c = ...
        y = torch.utils.checkpoint.checkpoint(lambda: (a + b) * c)

    PyTorch will not recompute ``... * c`` here,
    but it will recompute ``a + b``.
    We find this API more cumbersome to use and less flexible,
    because in many case, you know what you want to recompute, i.e. what you don't want to store.
    The PyTorch API is more about what you want to store, and then recompute everything else between.

    See also:
    https://github.com/rwth-i6/returnn/issues/1552
    https://discuss.pytorch.org/t/gradient-checkpointing/205416
    """

    def __init__(self):
        self.record_graph_scope = _RecordGraph()
        self.record_graph_scope.graph.gradient_checkpoint_scope_backref = self
        # Note: saved_tensors_hooks is thread local.
        self.saved_tensors_hooks_scope = torch.autograd.graph.saved_tensors_hooks(self._pack_hook, self._unpack_hook)
        self.entered = False
        self.entered_thread_ref = None
        self.exit_args: Optional[tuple] = None
        self.exited_saved_tensors_hooks_scope = False

    def __enter__(self):
        self.record_graph_scope.__enter__()
        self.saved_tensors_hooks_scope.__enter__()
        self.entered = True
        self.entered_thread_ref = ref(threading.current_thread())

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_args = (exc_type, exc_val, exc_tb)
        self.record_graph_scope.__exit__(exc_type, exc_val, exc_tb)
        if self.record_graph_scope.graph.is_any_recorded_tensor_alive():
            # Do not exit saved_tensors_hooks_scope here
            # because we still want to pack any tensors which were captured in our graph
            # by giving it a ref to the graph tensor.
            # However, we must track any further external calls to saved_tensors_hooks_scope,
            # to be able to properly remove it from the stack at the right point.
            _register_custom_saved_tensors_hooks(existing_scope=self.saved_tensors_hooks_scope)
            _register_custom_saved_tensors_hooks_thread_local_callback(
                _WeakMethod(self._custom_saved_tensors_hooks_callback, return_if_dead=False)
            )
        else:  # no relevant tensors alive anymore
            self.exit_saved_tensors_hooks_scope()

    def _maybe_exit_saved_tensors_hooks_scope(self):
        if self.exited_saved_tensors_hooks_scope:
            return
        if not self.exit_args:
            return
        # If we are in the right thread, maybe we can do the cleanup now.
        if self.entered_thread_ref() is threading.current_thread():
            if not self.record_graph_scope.graph.is_any_recorded_tensor_alive():
                self.exit_saved_tensors_hooks_scope()

    def __del__(self):
        if _python_exit:
            return
        # Note, be very careful what we do in __del__ because it might be called in a different thread!
        # Note that the __del__ will likely be called very late,
        # as the reference to the _Graph is kept alive until we used it for backprop,
        # as we keep this alive via _Graph.gradient_checkpoint_scope_backref
        # as long as any _GraphTensor is alive due to backprop pack_hook.
        self._maybe_exit_saved_tensors_hooks_scope()

    def exit_saved_tensors_hooks_scope(self):
        """
        exit saved_tensors_hooks_scope if not yet done.
        """
        assert self.entered_thread_ref() is threading.current_thread()
        if self.exit_args and not self.exited_saved_tensors_hooks_scope:
            # Note that via _register_custom_saved_tensors_hooks,
            # this saved_tensors_hooks_scope.__exit__ might get to our _custom_saved_tensors_hooks_exit below,
            # which will make sure that the order of __exit__ is correct.
            self.exited_saved_tensors_hooks_scope = True
            self.saved_tensors_hooks_scope.__exit__(*self.exit_args)

    def _pack_hook(self, x: torch.Tensor) -> Union[torch.Tensor, _GraphTensor]:
        if self.exit_args and not self.record_graph_scope.graph.is_any_recorded_tensor_alive():
            # No raw tensors alive anymore in graph_tensor_from_raw_tensor,
            # so we can exit saved_tensors_hooks_scope now.
            # (We might not always catch this properly in the Tensor _DelHook,
            #  e.g. when Tensor.__del__ runs in a different thread.)
            if _can_exit_saved_tensors_hooks_inside_hooks():
                self.exit_saved_tensors_hooks_scope()
            return x
        # _RecordGraph.__torch_dispatch__ should have recorded all newly created tensors.
        x_ = self.record_graph_scope.graph.graph_tensor_from_weak_raw_tensor.get(x, x)
        if isinstance(x_, _GraphTensor):
            x._RETURNN_grad_ckpt_del_hook = _DelHook(_WeakMethod(self._tensor_del_hook))
        return x_

    @staticmethod
    def _unpack_hook(x: Union[torch.Tensor, _GraphTensor]) -> torch.Tensor:
        if isinstance(x, _GraphTensor):
            if _can_exit_saved_tensors_hooks_inside_hooks():
                x.op.graph.gradient_checkpoint_scope_backref._maybe_exit_saved_tensors_hooks_scope()
            x.op.graph.maybe_recompute()
            return x.get_recomputed()
        return x

    def _tensor_del_hook(self):
        if _python_exit:
            return
        # Some of the relevant tensors got deleted.
        # If we are in the right thread, maybe we can do the cleanup now.
        self._maybe_exit_saved_tensors_hooks_scope()

    def _custom_saved_tensors_hooks_callback(self) -> bool:
        assert self.entered_thread_ref() is threading.current_thread()
        assert self.exit_args
        if self.record_graph_scope.graph.is_any_recorded_tensor_alive():
            return True  # keep callback alive
        else:
            self.exit_saved_tensors_hooks_scope()
            return False  # we are done, can delete callback


class _RecordGraph(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.graph = _Graph([])

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        graph = self.graph
        graph.maybe_store_rng_state(torch.device("cpu"))
        graph.maybe_store_amp_state(torch.device("cpu"))
        pytree.tree_map(graph.maybe_store_rng_state, args)
        pytree.tree_map(graph.maybe_store_rng_state, kwargs)
        out = func(*args, **kwargs)
        graph.record_op(func, args, kwargs, out)
        return out


@dataclass
class _Graph:
    ops_to_be_recomputed: List[_GraphOp] = field(default_factory=list)
    graph_tensor_from_weak_raw_tensor: WeakTensorKeyDictionary[torch.Tensor, _GraphTensor] = field(
        default_factory=WeakTensorKeyDictionary
    )
    stored_device_rng_states: Dict[torch.device, Any] = field(default_factory=dict)
    stored_device_amp_states: Dict[torch.device, Any] = field(default_factory=dict)
    # Keep scope alive as long as any _GraphTensor is alive due to backprop pack_hook.
    gradient_checkpoint_scope_backref: Optional[gradient_checkpoint_scope] = None

    def is_any_recorded_tensor_alive(self) -> bool:
        """
        :return: any recorded tensor is still alive.
            Recorded tensors are outputs from any ops which were recorded,
            i.e. ops under the gradient_checkpoint_scope.
        """
        # graph_tensor_from_weak_raw_tensor is a WeakTensorKeyDictionary,
        # i.e. once there is no other strong reference to some Tensor anymore,
        # it would also be removed from graph_tensor_from_weak_raw_tensor.
        return bool(self.graph_tensor_from_weak_raw_tensor)

    def record_op(self, func: Any, args: Sequence[Any], kwargs: Dict[str, Any], out: Any):
        """record op"""
        out_flat, _ = pytree.tree_flatten(out)
        wrapped_args = pytree.tree_map_only(torch.Tensor, self.maybe_map_raw_tensor_to_graph_tensor, args)
        wrapped_kwargs = pytree.tree_map_only(torch.Tensor, self.maybe_map_raw_tensor_to_graph_tensor, kwargs)
        op = _GraphOp(
            graph=self,
            func=func,
            args=wrapped_args,
            kwargs=wrapped_kwargs,
            out_flat_num=len(out_flat),
        )
        self.ops_to_be_recomputed.append(op)
        for i, out_flat_elem in enumerate(out_flat):
            if isinstance(out_flat_elem, torch.Tensor):
                if out_flat_elem in self.graph_tensor_from_weak_raw_tensor:
                    continue
                tensor_ = _GraphTensor(op=op, out_flat_idx=i)
                self.graph_tensor_from_weak_raw_tensor[out_flat_elem] = tensor_

    def maybe_store_rng_state(self, arg: Any):
        """
        Store RNG state if not yet stored for this device.
        We store it only once for the first usage,
        as we only restore it once for the recomputation,
        and then we rely on performing the recomputation in the correct order,
        which should be deterministic and lead to the same RNG output.
        """
        if isinstance(arg, torch.Tensor):
            device = arg.device
        elif isinstance(arg, torch.device):
            device = arg
        else:
            return
        if device not in self.stored_device_rng_states:
            self.stored_device_rng_states[device] = _get_dev_rng_state(device)

    def maybe_store_amp_state(self, arg: Any):
        """store AMP state if not yet stored for this device."""
        if isinstance(arg, torch.Tensor):
            device = arg.device
        elif isinstance(arg, torch.device):
            device = arg
        else:
            return
        if device not in self.stored_device_amp_states:
            self.stored_device_amp_states[device] = _get_dev_amp_state(device)

    def maybe_map_raw_tensor_to_graph_tensor(self, tensor: torch.Tensor) -> Union[_GraphTensor, torch.Tensor]:
        """raw tensor to graph tensor if available, otherwise return raw tensor."""
        return self.graph_tensor_from_weak_raw_tensor.get(tensor, tensor)

    def maybe_recompute(self):
        """
        Recompute.

        Make sure that the recomputations happen in the correct order,
        to get any random number generator state correct.

        Note that we considered to have an API here which allowed to only recompute a subset of the ops.
        It would still compute all from op idx 0 to some given op idx, but not the rest.
        On subsequent calls, it would then continue from the last idx until again the requested op idx.
        This works fine except of one important aspect: The RNG state.
        If there are any other ops in between which use the RNG state, the RNG state would not be correct anymore.
        To allow this, we then would need to get the RNG state again and reset it later again,
        which would add some further overhead.
        To keep things simple and to avoid this overhead, we recompute all ops together right now.

        However, we can at least remove the op from the list once it is computed.
        So once any referenced tensor is not needed anymore, it can be garbage collected.
        """
        if not self.ops_to_be_recomputed:
            return
        with _reset_rng_states_scope(self.stored_device_rng_states), _reset_amp_states_scope(
            self.stored_device_amp_states
        ):
            ops_reversed_queue = list(self.ops_to_be_recomputed)
            ops_reversed_queue.reverse()
            self.ops_to_be_recomputed.clear()
            while ops_reversed_queue:
                op = ops_reversed_queue.pop(-1)
                op.recompute()
        self.stored_device_rng_states.clear()
        self.stored_device_amp_states.clear()


@dataclass
class _GraphOp:
    graph: _Graph
    func: Any
    args: Optional[Sequence[Union[_GraphTensor, Any]]]
    kwargs: Optional[Dict[str, Union[_GraphTensor, Any]]]
    out_flat_num: int
    recomputed_out_flat: Optional[Sequence[torch.Tensor]] = None

    def recompute(self):
        """recompute, assuming all args are recomputed."""
        args = pytree.tree_map_only(_GraphTensor, _GraphTensor.get_recomputed, self.args)
        kwargs = pytree.tree_map_only(_GraphTensor, _GraphTensor.get_recomputed, self.kwargs)
        out = self.func(*args, **kwargs)
        out_flat, _ = pytree.tree_flatten(out)
        assert len(out_flat) == self.out_flat_num
        self.recomputed_out_flat = out_flat
        # potentially free any referenced resources. we don't need them anymore.
        self.args = None
        self.kwargs = None
        # self.func should be ok to keep, should ref some of the low-level aten functions


@dataclass
class _GraphTensor:
    op: _GraphOp
    out_flat_idx: int

    def get_recomputed(self) -> torch.Tensor:
        """assuming it was recomputed, return the raw tensor."""
        assert self.op.recomputed_out_flat is not None
        return self.op.recomputed_out_flat[self.out_flat_idx]


@contextlib.contextmanager
def _reset_rng_states_scope(states: Dict[torch.device, Any]):
    """
    Reset RNG states scope.
    Like torch.random.fork_rng but simpler.
    """
    prev_states = {dev: _get_dev_rng_state(dev) for dev in states.keys()}
    try:
        for dev, state in states.items():
            _set_dev_rng_state(dev, state)
        yield
    finally:
        for dev, state in prev_states.items():
            _set_dev_rng_state(dev, state)


def _get_dev_rng_state(dev: torch.device):
    if dev.type == "cpu":
        return torch.get_rng_state()
    dev_mod = getattr(torch, dev.type)
    return dev_mod.get_rng_state(dev)


def _set_dev_rng_state(dev: torch.device, state: Any):
    if dev.type == "cpu":
        torch.set_rng_state(state)
    else:
        dev_mod = getattr(torch, dev.type)
        dev_mod.set_rng_state(state, dev)


@contextlib.contextmanager
def _reset_amp_states_scope(states: Dict[torch.device, Any]):
    with contextlib.ExitStack() as stack:
        for dev, state in states.items():
            if not state:
                continue
            if dev.type == "cpu":
                stack.enter_context(torch.cpu.amp.autocast(**state))
            else:
                device_module = getattr(torch, dev.type)
                stack.enter_context(device_module.amp.autocast(**state))
        yield


def _get_dev_amp_state(dev: torch.device):
    if dev.type == "cpu":
        if not torch.is_autocast_cpu_enabled():
            return None
        return {
            "dtype": torch.get_autocast_cpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }

    if dev.type == "cuda":
        if not torch.is_autocast_enabled():
            return None
        return {
            "dtype": torch.get_autocast_gpu_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }

    device_module = getattr(torch, dev.type)
    if hasattr(device_module, "is_autocast_enabled") and hasattr(device_module, "get_autocast_dtype"):
        if not device_module.is_autocast_enabled():
            return None
        return {
            "dtype": device_module.get_autocast_dtype(),
            "cache_enabled": torch.is_autocast_cache_enabled(),
        }

    return None


class _DelHook:
    def __init__(self, callback):
        self.callback = callback

    def __del__(self):
        self.callback()


class _WeakMethod:
    # wrong type hint because mypy/PyCharm don't handle MethodType well
    def __init__(self, method: Union[MethodType, Callable], *, return_if_dead: Any = None):
        assert isinstance(method, MethodType)
        self.obj = ref(method.__self__)
        self.func = method.__func__
        self.return_if_dead = return_if_dead

    def __call__(self, *args, **kwargs):
        obj = self.obj()
        if obj is None:
            return self.return_if_dead
        return self.func(obj, *args, **kwargs)


def _can_exit_saved_tensors_hooks_inside_hooks() -> bool:
    """
    Check whether we can call torch.autograd.graph.saved_tensors_hooks.__exit__
    inside the pack or unpack hook itself.

    https://github.com/rwth-i6/returnn/issues/1581
    https://github.com/pytorch/pytorch/issues/130734
    """
    # TODO Return True in some later PyTorch version? Check https://github.com/pytorch/pytorch/issues/130734.
    return False


_orig_saved_tensors_hooks_enter = torch.autograd.graph.saved_tensors_hooks.__enter__
_orig_saved_tensors_hooks_exit = torch.autograd.graph.saved_tensors_hooks.__exit__
_custom_saved_tensors_hooks_tls_ctx = threading.local()
_custom_saved_tensors_hooks_lock = threading.Lock()  # only needed for non thread-locals, i.e. threads, methods
_custom_saved_tensors_hooks_registered_threads = WeakSet()


def _register_custom_saved_tensors_hooks(*, existing_scope: torch.autograd.graph.saved_tensors_hooks):
    """
    The purpose of our custom saved_tensors_hooks __enter__/__exit__ is to make sure that
    the order of __exit__ is correct, i.e. that we exit the scope in the correct order.

    See :func:`_custom_saved_tensors_hooks_enter` and :func:`_custom_saved_tensors_hooks_exit`.

    There is no need to call :func:`_unregister_custom_saved_tensors_hooks` later.
    It will be called automatically when the last scope is exited.
    """
    thread = threading.current_thread()
    with _custom_saved_tensors_hooks_lock:
        if thread in _custom_saved_tensors_hooks_registered_threads:
            return
        if getattr(_custom_saved_tensors_hooks_tls_ctx, "stack", None) is None:
            _custom_saved_tensors_hooks_tls_ctx.stack = []
            _custom_saved_tensors_hooks_tls_ctx.in_callback = False
            _custom_saved_tensors_hooks_tls_ctx.callbacks = []
            _custom_saved_tensors_hooks_tls_ctx.queued_exits = []
        _custom_saved_tensors_hooks_tls_ctx.active = True
        _custom_saved_tensors_hooks_tls_ctx.stack.append(existing_scope)
        _custom_saved_tensors_hooks_registered_threads.add(thread)
        if len(_custom_saved_tensors_hooks_registered_threads) == 1:
            torch.autograd.graph.saved_tensors_hooks.__enter__ = _custom_saved_tensors_hooks_enter
            torch.autograd.graph.saved_tensors_hooks.__exit__ = _custom_saved_tensors_hooks_exit


def _unregister_custom_saved_tensors_hooks():
    thread = threading.current_thread()
    with _custom_saved_tensors_hooks_lock:
        assert thread in _custom_saved_tensors_hooks_registered_threads
        assert (
            not _custom_saved_tensors_hooks_tls_ctx.stack
            and (_custom_saved_tensors_hooks_tls_ctx.in_callback or not _custom_saved_tensors_hooks_tls_ctx.callbacks)
            and not _custom_saved_tensors_hooks_tls_ctx.queued_exits
        ), (
            f"_unregister_custom_saved_tensors_hooks:"
            f" stack {_custom_saved_tensors_hooks_tls_ctx.stack},"
            f" in_callback {_custom_saved_tensors_hooks_tls_ctx.in_callback},"
            f" callbacks {_custom_saved_tensors_hooks_tls_ctx.callbacks},"
            f" queued_exits {_custom_saved_tensors_hooks_tls_ctx.queued_exits}"
        )
        _custom_saved_tensors_hooks_tls_ctx.active = False
        _custom_saved_tensors_hooks_registered_threads.remove(thread)
        if not _custom_saved_tensors_hooks_registered_threads:
            torch.autograd.graph.saved_tensors_hooks.__enter__ = _orig_saved_tensors_hooks_enter
            torch.autograd.graph.saved_tensors_hooks.__exit__ = _orig_saved_tensors_hooks_exit


def _custom_saved_tensors_hooks_enter(self: torch.autograd.graph.saved_tensors_hooks):
    _custom_saved_tensors_hooks_call_callbacks()
    # The callbacks might have unregistered us. Only add to the stack if we are still active.
    if _custom_saved_tensors_hooks_tls_ctx.active:
        _custom_saved_tensors_hooks_tls_ctx.stack.append(self)
    return _orig_saved_tensors_hooks_enter(self)


def _custom_saved_tensors_hooks_exit(
    self: torch.autograd.graph.saved_tensors_hooks, exc_type=None, exc_val=None, exc_tb=None
):
    if self not in _custom_saved_tensors_hooks_tls_ctx.stack:
        raise Exception(
            f"saved_tensors_hooks __exit__ mismatch."
            f" stack {_custom_saved_tensors_hooks_tls_ctx.stack},"
            f" queued_exits {_custom_saved_tensors_hooks_tls_ctx.queued_exits},"
            f" got self {self}"
        )
    _custom_saved_tensors_hooks_tls_ctx.queued_exits.append(self)
    _custom_saved_tensors_hooks_call_callbacks()
    while _custom_saved_tensors_hooks_tls_ctx.stack:
        scope = _custom_saved_tensors_hooks_tls_ctx.stack[-1]
        if scope not in _custom_saved_tensors_hooks_tls_ctx.queued_exits:
            # Need to wait for this scope to exit first.
            # Once we exit it, we would then exit also the others when they are on top.
            break
        _custom_saved_tensors_hooks_tls_ctx.stack.pop(-1)
        _custom_saved_tensors_hooks_tls_ctx.queued_exits.remove(scope)
        _orig_saved_tensors_hooks_exit(scope, exc_type, exc_val, exc_tb)
        exc_type, exc_val, exc_tb = None, None, None  # do not propagate this again (even though it's ignored anyway)
    if not _custom_saved_tensors_hooks_tls_ctx.stack:
        assert not _custom_saved_tensors_hooks_tls_ctx.queued_exits
        if _custom_saved_tensors_hooks_tls_ctx.active:  # might have been unregistered in the meantime by callbacks
            _unregister_custom_saved_tensors_hooks()


def _register_custom_saved_tensors_hooks_thread_local_callback(cb: Callable[[], bool]):
    """
    Register some thread-local callback function which is called on saved_tensors_hooks __enter__ and __exit__.
    If it returns True, it is kept alive, otherwise removed.
    """
    assert not _custom_saved_tensors_hooks_tls_ctx.in_callback
    _custom_saved_tensors_hooks_tls_ctx.callbacks.append(cb)


def _custom_saved_tensors_hooks_call_callbacks():
    if _custom_saved_tensors_hooks_tls_ctx.in_callback:
        return  # avoid recursive calls
    try:
        _custom_saved_tensors_hooks_tls_ctx.in_callback = True
        _custom_saved_tensors_hooks_tls_ctx.callbacks = [
            cb for cb in _custom_saved_tensors_hooks_tls_ctx.callbacks if cb()
        ]
        if not _custom_saved_tensors_hooks_tls_ctx.active:  # was cleaned up by some of the callbacks
            assert not _custom_saved_tensors_hooks_tls_ctx.callbacks and not _custom_saved_tensors_hooks_tls_ctx.stack
    finally:
        _custom_saved_tensors_hooks_tls_ctx.in_callback = False


def _python_exit_handler():
    global _python_exit
    _python_exit = True


_python_exit = False
atexit.register(_python_exit_handler)
