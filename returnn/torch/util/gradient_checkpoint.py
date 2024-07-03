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
from dataclasses import dataclass, field
import contextlib
from weakref import ref, WeakSet
import threading

import torch
from torch.utils.weak import WeakTensorKeyDictionary  # needs Torch >=2.0.0

# noinspection PyProtectedMember
from torch.utils._python_dispatch import TorchDispatchMode

# noinspection PyProtectedMember
import torch.utils._pytree as pytree


__all__ = ["gradient_checkpoint_scope"]


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
        if self.record_graph_scope.graph.graph_tensor_from_raw_tensor:
            # Do not exit saved_tensors_hooks_scope here
            # because we still want to pack any tensors which were captured in our graph
            # by giving it a ref to the graph tensor.
            # However, we must track any further external calls to saved_tensors_hooks_scope,
            # to be able to properly remove it from the stack at the right point.
            _register_custom_saved_tensors_hooks(existing_scope=self.saved_tensors_hooks_scope)
            _register_custom_saved_tensors_hooks_thread_local_callback(self._custom_saved_tensors_hooks_callback)
        else:  # no relevant tensors alive anymore
            self.exit_saved_tensors_hooks_scope()

    def _maybe_exit_saved_tensors_hooks_scope(self):
        if self.exited_saved_tensors_hooks_scope:
            return
        if not self.exit_args:
            return
        # If we are in the right thread, maybe we can do the cleanup now.
        if self.entered_thread_ref() is threading.current_thread():
            if not self.record_graph_scope.graph.graph_tensor_from_raw_tensor:
                self.exit_saved_tensors_hooks_scope()

    def __del__(self):
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
            self.saved_tensors_hooks_scope.__exit__(*self.exit_args)
            self.exited_saved_tensors_hooks_scope = True

    def _pack_hook(self, x: torch.Tensor) -> Union[torch.Tensor, _GraphTensor]:
        if self.exit_args and not self.record_graph_scope.graph.graph_tensor_from_raw_tensor:
            # No raw tensors alive anymore in graph_tensor_from_raw_tensor,
            # so we can exit saved_tensors_hooks_scope now.
            self.exit_saved_tensors_hooks_scope()
            return x
        x_ = self.record_graph_scope.graph.graph_tensor_from_raw_tensor.get(x, x)
        if isinstance(x_, _GraphTensor):
            x._RETURNN_grad_ckpt_del_hook = _DelHook(self._tensor_del_hook)
        return x_

    @staticmethod
    def _unpack_hook(x: Union[torch.Tensor, _GraphTensor]) -> torch.Tensor:
        if isinstance(x, _GraphTensor):
            x.op.graph.gradient_checkpoint_scope_backref._maybe_exit_saved_tensors_hooks_scope()
            x.op.graph.recompute()
            return x.get_recomputed()
        return x

    def _tensor_del_hook(self):
        # Some of the relevant tensors got deleted.
        # If we are in the right thread, maybe we can do the cleanup now.
        self._maybe_exit_saved_tensors_hooks_scope()

    def _custom_saved_tensors_hooks_callback(self) -> bool:
        assert self.entered_thread_ref() is threading.current_thread()
        assert self.exit_args
        if self.record_graph_scope.graph.graph_tensor_from_raw_tensor:
            return True  # keep callback alive
        else:
            self.exit_saved_tensors_hooks_scope()
            return False  # we are done


class _RecordGraph(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.graph = _Graph([])

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        out = func(*args, **kwargs)
        self.graph.record_op(func, args, kwargs, out)
        return out


@dataclass
class _Graph:
    ops: List[_GraphOp] = field(default_factory=list)
    graph_tensor_from_raw_tensor: WeakTensorKeyDictionary[torch.Tensor, _GraphTensor] = field(
        default_factory=WeakTensorKeyDictionary
    )
    stored_device_rng_states: Dict[torch.device, Any] = field(default_factory=dict)
    stored_device_amp_states: Dict[torch.device, Any] = field(default_factory=dict)
    # Keep scope alive as long as any _GraphTensor is alive due to backprop pack_hook.
    gradient_checkpoint_scope_backref: Optional[gradient_checkpoint_scope] = None

    def record_op(self, func: Any, args: Sequence[Any], kwargs: Dict[str, Any], out: Any):
        """record op"""
        self.maybe_store_rng_state(torch.device("cpu"))
        self.maybe_store_amp_state(torch.device("cpu"))
        pytree.tree_map(self.maybe_store_rng_state, args)
        pytree.tree_map(self.maybe_store_rng_state, kwargs)
        wrapped_args = pytree.tree_map_only(torch.Tensor, self.maybe_map_raw_tensor_to_graph_tensor, args)
        wrapped_kwargs = pytree.tree_map_only(torch.Tensor, self.maybe_map_raw_tensor_to_graph_tensor, kwargs)
        out_flat = pytree.tree_flatten(out)
        op = _GraphOp(
            graph=self,
            op_idx=len(self.ops),
            func=func,
            args=wrapped_args,
            kwargs=wrapped_kwargs,
            out_flat_num=len(out_flat),
        )
        self.ops.append(op)
        for i, out_flat_elem in enumerate(out_flat):
            if isinstance(out_flat_elem, torch.Tensor):
                if out_flat_elem in self.graph_tensor_from_raw_tensor:
                    continue
                tensor_ = _GraphTensor(op=op, out_flat_idx=i)
                self.graph_tensor_from_raw_tensor[out_flat_elem] = tensor_

    def maybe_store_rng_state(self, arg: Any):
        """store RNG state if not yet stored for this device."""
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
        return self.graph_tensor_from_raw_tensor.get(tensor, tensor)

    def recompute(self):
        """
        Recompute.

        Make sure that the recomputations happen in the correct order,
        to get any random number generator state correct.
        """
        with _reset_rng_states_scope(self.stored_device_rng_states), _reset_amp_states_scope(
            self.stored_device_amp_states
        ):
            for op in self.ops:
                op.recompute()


@dataclass
class _GraphOp:
    graph: _Graph
    op_idx: int
    func: Any
    args: Sequence[Union[_GraphTensor, Any]]
    kwargs: Dict[str, Union[_GraphTensor, Any]]
    out_flat_num: int
    recomputed_out_flat: Optional[Sequence[torch.Tensor]] = None

    def recompute(self):
        """recompute, assuming all args are recomputed."""
        args = pytree.tree_map_only(_GraphTensor, _GraphTensor.get_recomputed, self.args)
        kwargs = pytree.tree_map_only(_GraphTensor, _GraphTensor.get_recomputed, self.kwargs)
        out_flat = self.func(*args, **kwargs)
        assert len(out_flat) == self.out_flat_num
        self.recomputed_out_flat = out_flat


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
            and not _custom_saved_tensors_hooks_tls_ctx.callbacks
            and not _custom_saved_tensors_hooks_tls_ctx.queued_exits
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


def _custom_saved_tensors_hooks_exit(self: torch.autograd.graph.saved_tensors_hooks, exc_type, exc_val, exc_tb):
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
    _custom_saved_tensors_hooks_tls_ctx.callbacks.append(cb)


def _custom_saved_tensors_hooks_call_callbacks():
    if _custom_saved_tensors_hooks_tls_ctx.in_callback:
        return  # avoid recursive calls
    try:
        _custom_saved_tensors_hooks_tls_ctx.in_callback = True
        _custom_saved_tensors_hooks_tls_ctx.callbacks = [
            cb for cb in _custom_saved_tensors_hooks_tls_ctx.callbacks if cb()
        ]
    finally:
        _custom_saved_tensors_hooks_tls_ctx.in_callback = False
