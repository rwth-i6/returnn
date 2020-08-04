
"""
This provides lazy module wrappers such that old-style RETURNN-as-a-framework imports are still working.
E.g. like::

    import returnn.TFUtil
    import returnn.TFNativeOp

"""

import sys
import os
import types
import typing
import importlib

old_to_new_mod_mapping = {
  "Util": "util.basic",
  "TFUtil": "tf.util.basic",
  "TFNativeOp": "tf.native_op",
  "TFNetworkLayer": "tf.layers.basic",
  # ...
}

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_mod_cache = {}  # new/old mod name -> mod


def setup(package_name=__package__, modules=None):
  """
  This does the setup, such that all the modules become available in the `returnn` package.
  It does not import all the modules now, but instead provides them lazily.

  :param str package_name: "returnn" by default
  :param dict[str,types.ModuleType]|None modules: if set, will do ``modules[old_mod_name] = mod``
  """
  for old_mod_name, new_mod_name in sorted(old_to_new_mod_mapping.items()):
    full_mod_name = "returnn.%s" % new_mod_name
    full_old_mod_name = "%s.%s" % (package_name, old_mod_name)
    if full_mod_name in _mod_cache:
      mod = _mod_cache[full_mod_name]
    elif full_mod_name in sys.modules:
      mod = sys.modules[full_mod_name]
      _mod_cache[full_mod_name] = mod
    else:
      mod = _LazyLoader(
        full_mod_name=full_mod_name,
        full_old_mod_name=full_old_mod_name, old_mod_name=old_mod_name, modules=modules)
    if old_mod_name not in sys.modules:
      sys.modules[old_mod_name] = mod
    if full_old_mod_name not in sys.modules:
      sys.modules[full_old_mod_name] = mod
    if modules is not None:
      modules[old_mod_name] = mod


class _LazyLoader(types.ModuleType):
  """
  Lazily import a module, mainly to avoid pulling in large dependencies.
  Code borrowed from TensorFlow, and simplified, and extended.
  """
  def __init__(self, full_mod_name, **kwargs):
    """
    :param str full_mod_name:
    """
    super(_LazyLoader, self).__init__(full_mod_name)
    fn = "%s/%s.py" % (_base_dir, full_mod_name.replace(".", "/"))
    if not os.path.exists(fn):
      fn = "%s/%s/__init__.py" % (_base_dir, full_mod_name.replace(".", "/"))
      assert os.path.exists(fn), "_LazyLoader: mod %r not found in %r" % (full_mod_name, _base_dir)
    self.__file__ = fn
    self._lazy_mod_config = dict(full_mod_name=full_mod_name, **kwargs)  # type: typing.Dict[str]

  def _load(self):
    full_mod_name = self.__name__
    lazy_mod_config = self._lazy_mod_config
    old_mod_name = lazy_mod_config.get("old_mod_name", None)
    full_old_mod_name = lazy_mod_config.get("full_old_mod_name", None)
    modules = lazy_mod_config.get("modules", None)
    if full_mod_name in _mod_cache:
      module = _mod_cache[full_mod_name]
    else:
      try:
        module = importlib.import_module(full_mod_name)
      except Exception:
        # Note: If we get any exception in the module itself (e.g. No module named 'theano' or so),
        # just pass it on. But this can happen.
        raise
      _mod_cache[full_mod_name] = module

    if old_mod_name:
      sys.modules[old_mod_name] = module
    if full_old_mod_name:
      sys.modules[full_old_mod_name] = module
    if modules is not None:
      assert old_mod_name
      modules[old_mod_name] = module

    # Do not set self.__dict__, because the module itself could later update itself.
    return module

  def __getattribute__(self, item):
    # Implement also __getattribute__ such that early access to just self.__dict__ (e.g. via vars(self)) also works.
    if item == "__dict__":
      # noinspection PyBroadException
      try:
        mod = self._load()
      except Exception:  # many things could happen
        print("WARNING: %s cannot be imported, __dict__ not available" % self.__name__)
        # In many cases, this is not so critical, because we likely just checked the dict content or so.
        # This should be safe, as we have this registered in sys.modules, and some code just iterates
        # through all sys.modules to check for something.
        # Any other attribute access will lead to the real exception.
        # We ignore this for __dict__, and just return a dummy empty dict.
        return {}
      return getattr(mod, "__dict__")
    return super(_LazyLoader, self).__getattribute__(item)

  def __getattr__(self, item):
    module = self._load()
    return getattr(module, item)

  def __dir__(self):
    module = self._load()
    return dir(module)

  def __setattr__(self, key, value):
    if key in ["__file__", "_lazy_mod_config"]:
      super(_LazyLoader, self).__setattr__(key, value)
      return
    module = self._load()
    setattr(module, key, value)
