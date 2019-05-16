
"""
This is here so that whole RETURNN can be imported as a submodule.
"""

# Currently all of RETURNN is written with absolute imports.
# This probably should be changed, see also: https://github.com/rwth-i6/returnn/issues/162
# If we change that to relative imports, for some of the scripts we might need some solution like this:
# https://stackoverflow.com/questions/54576879/

# Anyway, as a very ugly workaround, to make it possible to import RETURNN as a package, we have this hack:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import types as _types
import os as _os


if globals().get("__package__", None) is None:
  __package__ = __name__  # https://www.python.org/dev/peps/pep-0366/


_my_dir = _os.path.dirname(_os.path.abspath(__file__))
_mod_cache = {}  # mod_name -> mod


def _setup():
  """
  This does the setup, such that all the modules become available in the `returnn` package.
  It does not import all the modules now, but instead provides them lazily.
  """
  import os
  import sys

  for fn in sorted(os.listdir(_my_dir)):
    mod_name, ext = os.path.splitext(os.path.basename(fn))
    if ext != ".py":
      continue
    if mod_name.startswith("__"):
      continue
    if mod_name in sys.modules:
      # This is difficult to get right.
      # We will just use the existing module. Print a warning.
      print("RETURNN import warning: module %r already imported as an absolute module" % mod_name)
      mod = sys.modules[mod_name]
    else:
      mod = _LazyLoader(mod_name)
    globals()[mod_name] = mod  # make available as `returnn.<mod_name>` (attribute)
    sys.modules[mod_name] = mod  # absolute import
    sys.modules["%s.%s" % (__package__, mod_name)] = mod  # `returnn.<mod_name>` as relative import


class _LazyLoader(_types.ModuleType):
  """
  Lazily import a module, mainly to avoid pulling in large dependencies.
  Code borrowed from TensorFlow, and simplified, and extended.
  """
  def __init__(self, name):
    super(_LazyLoader, self).__init__(name)
    self.__file__ = "%s/%s.py" % (_my_dir, name)

  def _load(self):
    name = self.__name__
    if name in _mod_cache:
      return _mod_cache[name]
    # This assert can be confusing. But: This module instance itself might become the new imported module.
    assert "." not in name
    import sys
    import importlib
    full_mod_name = "%s.%s" % (__package__, name)
    sys.modules.pop(full_mod_name, None)  # Make sure that we really load it.
    try:
      module = importlib.import_module("." + name, __package__)  # relative import
    except Exception:
      # Note: If we get any exception in the module itself (e.g. No module named 'theano' or so),
      # just pass it on. But this can happen.
      sys.modules[full_mod_name] = self  # Make sure that the next import again reaches us again.
      raise
    _mod_cache[name] = module
    _mod_cache[full_mod_name] = module
    sys.modules[name] = module  # shortcut for absolute import
    sys.modules[full_mod_name] = module  # shortcut for relative import
    globals()[name] = module  # shortcut the lazy loader
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
    if key in ["__file__"]:
      super(_LazyLoader, self).__setattr__(key, value)
      return
    module = self._load()
    setattr(module, key, value)


_setup()

