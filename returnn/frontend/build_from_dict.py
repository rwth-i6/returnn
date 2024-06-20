"""
Construct modules (or other objects) from dictionaries.
"""

from __future__ import annotations
from typing import Union, Any, Type, Dict
import importlib
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
    :return: cls(*args, **d, **kwargs)
    """
    if "class" not in d:
        raise ValueError(f"build_from_dict: Missing 'class' key in dict: {d}")
    d = d.copy()
    cls_name = d.pop("class")
    cls = _get_cls(cls_name)
    return cls(*args, **d, **kwargs)


def build_dict(cls: Type, **kwargs) -> Dict[str, Any]:
    """
    Build a dictionary for :func:`build_from_dict`.
    The class name is stored in the `"class"` key.

    Note that this is intended to be used for serialization
    and also to get a unique stable hashable representation
    (e.g. for Sisyphus :func:`sis_hash_helper`)
    which should not change if the class is renamed or moved
    to keep the hash stable.
    """
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
