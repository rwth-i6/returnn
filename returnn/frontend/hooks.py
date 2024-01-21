"""
Hooks logic
"""

from __future__ import annotations
from typing import Optional, Any, Dict, Callable, TypeVar, Tuple
from types import MethodType
from collections import OrderedDict
import weakref


__all__ = ["setup_post_hook_on_method", "RemovableHandle"]


T = TypeVar("T")


def setup_post_hook_on_method(
    obj: Any,
    attr: str,
    hook: Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]],
    *,
    prepend: bool = False,
) -> RemovableHandle:
    """
    :param obj: e.g. module
    :param attr: attribute name to get the method, via ``getattr(obj, attr)``
    :param hook:
    :param prepend: if there are multiple hooks, this will be registered in front of all, otherwise at the end
    """
    method = MethodWithHooks.get(obj, attr)
    return _register_new_hook(method, method.post_hooks, hook, prepend=prepend)


class MethodWithHooks:
    """
    Method with hooks

    (Not sure yet whether this should be exposed in the API...)
    """

    @classmethod
    def get(cls, obj: Any, attr: str) -> MethodWithHooks:
        """get existing or init new :class:`MethodWithHooks`"""
        method = getattr(obj, attr)
        if not isinstance(method, MethodWithHooks):
            assert (
                isinstance(method, MethodType)
                and method.__self__ is obj
                and method.__func__ is getattr(obj.__class__, attr)
            ), (
                f"{obj}.{attr} is {method}, {method.__self__} is not obj"
                f" or {method.__func__} is not {getattr(obj.__class__, attr)}"
            )  # not necessary, just what we expect here...
            method = MethodWithHooks(obj, attr)
            method.setup()
        return method

    def __init__(self, obj: Any, attr: str):
        """
        :param obj:
        :param attr:
        """
        self.obj = obj
        self.attr = attr
        self.post_hooks: Dict[int, Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]]] = OrderedDict()

    def setup(self):
        """setup on object"""
        setattr(self.obj, self.attr, self)
        if self.attr == "__call__":
            self.obj.__class__.__call__ = _CallWrapperClass(self.obj.__class__)

    def uninstall(self):
        """undo the setup"""
        assert not self.post_hooks
        delattr(self.obj, self.attr)
        if self.attr == "__call__":
            call_wrapper = self.obj.__class__.__call__
            assert isinstance(call_wrapper, _CallWrapperClass)
            self.obj.__class__.__call__ = call_wrapper.orig_call

    def has_any_hook(self) -> bool:
        """whether there is any hook"""
        return bool(self.post_hooks)

    def __call__(self, *args, **kwargs):
        function = getattr(self.obj.__class__, self.attr)
        if isinstance(function, _CallWrapperClass):
            function = function.orig_call
        result = function(self.obj, *args, **kwargs)
        for hook in self.post_hooks.values():
            result = hook(self.obj, args, kwargs, result)
        return result


class _CallWrapperClass:
    """
    We need special handling for ``obj(*args, **kwargs)`` - simply wrapping ``__call__`` is not enough.
    An instance of this class here would overwrite the obj.__class__.__call__.
    """

    def __init__(self, cls):
        self.cls = cls
        self.orig_call = getattr(cls, "__call__")

    def __call__(self, obj, *args, **kwargs):  # called on the class
        call_func = getattr(obj, "__call__")
        if isinstance(call_func, MethodWithHooks):  # obj has hooks installed
            return call_func(*args, **kwargs)
        # This is what we would expect currently, but the user might have dome some weird things,
        # e.g. overwritten obj.__call__ or obj.__class__.__call__ in some other way,
        # so maybe extend this then, if we encounter this.
        # (Only do it when we really now the case properly, and add a test case for it.)
        assert isinstance(call_func, MethodType)
        assert call_func.__func__ is self
        assert call_func.__self__ is obj
        return self.orig_call(obj, *args, **kwargs)

    def __get__(self, obj, objtype=None):
        if obj is not None:
            return MethodType(self, obj)
        return self


def _register_new_hook(
    method: MethodWithHooks,
    hooks_dict: Dict[int, Any],
    hook: Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]],
    *,
    prepend: bool = False,
) -> RemovableHandle:
    """register"""
    handle = RemovableHandle(method, hooks_dict)
    hooks_dict[handle.id] = hook
    if prepend:
        assert isinstance(hooks_dict, OrderedDict)
        hooks_dict.move_to_end(handle.id, last=False)
    return handle


class RemovableHandle:
    """
    A handle which provides the capability to remove a hook.

    Code partly copied from PyTorch.
    """

    id: int
    next_id: int = 0

    def __init__(self, method: MethodWithHooks, hooks_dict: Dict[int, Any]):
        self.method_ref = weakref.ref(method)
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

    def remove(self) -> None:
        """remove hook"""
        method: MethodWithHooks = self.method_ref()
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]
            if method and not method.has_any_hook():  # we removed the last hook
                method.uninstall()
