"""
Construct modules (or other objects) from dictionaries.
"""

from __future__ import annotations
from types import FunctionType, BuiltinFunctionType
from typing import Union, Any, Type, Callable, Dict
import importlib
import functools
import returnn.frontend as rf


__all__ = ["build_from_dict", "build_dict"]


def build_from_dict(d: Dict[str, Any], *args, **kwargs) -> Union[rf.Module, Any]:
    """
    Build a module (or other object) from a dictionary.
    `"class"` in the dict is required and specifies the class to be instantiated.
    The other options are passed to the class constructor.

    :param d: dictionary with the class name and other options
    :param args: passed to the class constructor
    :param kwargs: passed to the class constructor
    :return: ``cls(*args, **d, **kwargs)`` (after "class" was popped from ``d``)
    """
    if "class" not in d:
        raise ValueError(f"build_from_dict: Missing 'class' key in dict: {d}")
    d = d.copy()
    cls_name = d.pop("class")
    cls = _get_cls(cls_name)
    if isinstance(cls, (FunctionType, BuiltinFunctionType)):
        if args or kwargs or d:
            return functools.partial(cls, *args, **d, **kwargs)
        return cls
    if not isinstance(cls, type):
        raise ValueError(f"build_from_dict: Expected class, got {cls!r} for class_name {cls_name!r}")
    return cls(*args, **d, **kwargs)


def build_dict(cls: Union[Type, FunctionType, BuiltinFunctionType, Callable[..., Any]], **kwargs) -> Dict[str, Any]:
    """
    Build a dictionary for :func:`build_from_dict`.
    The class name is stored in the `"class"` key.

    Note that this is intended to be used for serialization
    and also to get a unique stable hashable representation
    (e.g. for Sisyphus :func:`sis_hash_helper`)
    which should not change if the class is renamed or moved
    to keep the hash stable.

    :param cls: class or function. actually we do not expect any other Callable here.
        It's just Callable in the type hint because MyPy/PyCharm don't correctly support FunctionType
            (https://github.com/python/mypy/issues/3171).
    :param kwargs: other kwargs put into the dict. expect to contain only other serializable values.
    :return: build dict. can easily be serialized. to be used with :func:`build_from_dict`.
    """
    if not isinstance(cls, (type, FunctionType, BuiltinFunctionType)):
        raise TypeError(f"build_dict: Expected class or function, got: {cls!r}")
    return {"class": _get_cls_name(cls), **kwargs}


def _get_cls(cls_name: str) -> Type:
    if "." not in cls_name:
        raise ValueError(f"Expected '.' in class name: {cls_name}")
    mod_name, cls_name = cls_name.rsplit(".", 1)
    if mod_name == "rf":
        return getattr(rf, cls_name)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def _get_cls_name(cls: Type) -> str:
    if getattr(rf, cls.__name__, None) is cls:
        return f"rf.{cls.__name__}"
    return f"{cls.__module__}.{cls.__name__}"
