"""
Hot reloading
"""

from __future__ import annotations
import os
import sys
from typing import Optional, Any, Dict, Tuple, Set, List, Callable
from types import FunctionType
from functools import partial
from importlib import reload
from returnn.config import Config, get_global_config
from returnn.util.better_exchook import debug_shell


def should_use_hot_reloading(*, config: Optional[Config] = None) -> bool:
    """
    Check if hot reloading should be used.
    """
    if not sys.stdin.isatty():
        return False
    if not config:
        config = get_global_config(raise_exception=False)
    if config and config.bool("use_hot_reloading", False):
        return True
    return os.environ.get("RETURNN_USE_HOT_RELOADING", "0") == "1"


class ConfigHotReloader:
    """
    A class that can hot reload the values in a config.
    """

    def __init__(self, config: Dict[str, Any]):
        print("Initializing hot reloader on config...")
        self.config = config
        self._modules, self._relevant_keys = _find_modules_in_config(config)
        self._mod_mtimes = {mod_name: os.path.getmtime(sys.modules[mod_name].__file__) for mod_name in self._modules}

    def any_module_changed(self) -> bool:
        """
        Check if any of the modules have changed.
        """
        for mod_name in self._modules:
            current_mtime = os.path.getmtime(sys.modules[mod_name].__file__)
            if current_mtime != self._mod_mtimes[mod_name]:
                return True
        return False

    def reload_changed_modules(self):
        """
        Reload any changed modules and update the config.
        """
        for mod_name in self._modules:
            current_mtime = os.path.getmtime(sys.modules[mod_name].__file__)
            if current_mtime != self._mod_mtimes[mod_name]:
                print(f"Module '{mod_name}' has changed, reloading...")
                reload(sys.modules[mod_name])
                self._mod_mtimes[mod_name] = current_mtime

        for k in self._relevant_keys:
            _iter(self.config[k], update_func=partial(self.config.__setitem__, k))

    def wait_for_user(self):
        """
        If the user can interact, wait for the user to press Enter.
        """
        exc_type, exc, exc_traceback = sys.exc_info()
        assert sys.stdin.isatty()
        while True:
            choices = {"r": "reload modules and try again"}
            if exc is not None:
                choices["d"] = "debug"
                choices["e"] = "reraise"
            choices["s"] = "shell"
            choices["q"] = "quit"
            answer = input("Hot reloading: " + ", ".join(f"'{k}' to {v}" for k, v in choices.items()) + ": ")
            if answer == "r":
                if self.any_module_changed():
                    return
                print("No changes detected? Please change the source code:", self._modules)
            elif answer == "d" and exc is not None:
                debug_shell({}, {}, traceback=exc_traceback)
            elif answer == "e" and exc is not None:
                raise exc.with_traceback(exc_traceback)
            elif answer == "s":
                debug_shell({}, {})
            elif answer == "q":
                sys.exit(1)
            else:
                print(f"Invalid choice {answer!r}")


def hot_reload_config(config: Dict[str, Any]):
    """
    Hot reload the values in the config,
    i.e. if the value references a class or function, it will be reloaded from the source code.
    """
    # First collect all relevant modules.
    collected_modules, relevant_keys = _find_modules_in_config(config)

    # Now reload all collected modules.
    for mod_name in collected_modules:
        print(f"Reloading module '{mod_name}'...")
        reload(sys.modules[mod_name])

    # Finally, update the config values to point to the reloaded objects.
    for k in relevant_keys:
        _iter(config[k], update_func=partial(config.__setitem__, k))


def _find_modules_in_config(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    visited_modules = set()
    collected_modules = []
    relevant_keys = []
    for k, v in config.items():
        # get module and name from the value
        state = _IterState()
        _iter(v, state=state)
        if state.collected_modules:
            relevant_keys.append(k)
            print(f"Found modules for config key '{k}': {state.collected_modules}")
            for mod_name in state.collected_modules:
                if mod_name not in visited_modules:
                    visited_modules.add(mod_name)
                    collected_modules.append(mod_name)
    return collected_modules, relevant_keys


def _iter(
    obj: Any,
    *,
    state: Optional[_IterState] = None,
    update_func: Optional[Callable] = None,
) -> Any:
    """
    Find all modules that are referenced by the object,
    i.e. the module of the object itself and the modules of its attributes.
    """
    if state is None:
        state = _IterState()
    if obj is None:
        return False
    if isinstance(obj, (int, float, str, bool)):
        return False
    if isinstance(obj, list):
        have_update = False
        for i, item in enumerate(obj):
            have_update &= _iter(item, update_func=partial(obj.__setitem__, i) if update_func else None, state=state)
        return have_update
    if isinstance(obj, (tuple, set)):
        res = list(obj)
        have_update = False
        for i, item in enumerate(res):
            have_update &= _iter(item, update_func=partial(res.__setitem__, i) if update_func else None, state=state)
        if have_update and update_func:
            update_func(type(obj)(res))
        return have_update
    if isinstance(obj, dict):
        have_update = False
        for k, v in obj.items():
            have_update &= _iter(v, update_func=partial(obj.__setitem__, k) if update_func else None, state=state)
        return have_update
    if isinstance(obj, partial):
        have_update = False
        new_opts = {}
        for attr in ["func", "args", "keywords"]:
            new_opts[attr] = getattr(obj, attr)
            have_update &= _iter(
                getattr(obj, attr),
                update_func=partial(new_opts.__setitem__, attr) if update_func else None,
                state=state,
            )
        if have_update and update_func:
            update_func(partial(new_opts["func"], *new_opts["args"], **new_opts["keywords"]))
        return have_update
    mod_name, name = _get_module_name_and_name_from_obj(obj)
    if mod_name is not None and _is_custom_module(mod_name):
        if mod_name not in state.visited_modules:
            state.visited_modules.add(mod_name)
            state.collected_modules.append(mod_name)
        if update_func:
            # assuming it is already reloaded
            mod = sys.modules[mod_name]
            obj_ = getattr(mod, name)
            update_func(obj_)
            return True
    return False


class _IterState:
    def __init__(self):
        self.visited_modules: Set[str] = set()
        self.collected_modules: List[str] = []


def _get_module_name_and_name_from_obj(obj: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Get the module name and the name of the object.
    """
    if isinstance(obj, FunctionType):
        return obj.__module__, obj.__name__
    elif isinstance(obj, type):
        return obj.__module__, obj.__name__
    else:
        return None, None


def _is_custom_module(module_name: Optional[str]) -> bool:
    """
    Check if the module is a custom module, i.e. not a built-in module or a standard library module.
    Assumes that the module is already imported and available in sys.modules.
    """
    if module_name in sys.builtin_module_names:
        return False
    if module_name in getattr(sys, "stdlib_module_names", set()):
        return False
    mod = sys.modules[module_name]
    mod_filename: Optional[str] = getattr(mod, "__file__", None)
    if mod_filename is None:
        # built-in module
        return False
    if mod_filename.startswith(sys.base_prefix):
        return False
    if mod_filename.startswith(sys.prefix):
        return False
    if mod_filename.startswith(sys.exec_prefix):
        return False
    assert mod_filename.endswith(".py")
    return True
