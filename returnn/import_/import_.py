"""
This provides the main :func:`import_` function.
"""

import types
import importlib
from .git import get_repo_path
from .common import module_name


def import_(repo, path, version=None):
  """
  :param str repo: e.g. "github.com/rwth-i6/returnn-experiments"
  :param str path: path inside the repo, without starting "/"
  :param str|None version: e.g. "20211231-0123abcd0123". None for development working copy
  :rtype: object|types.ModuleType
  """
  assert path and path[:1] != "/" and ".." not in path
  repo_path = get_repo_path(repo=repo, version=version)
  # `module_name` has the side effect that `import_module` below will just work.
  mod_name = module_name(repo=repo, repo_path=repo_path, path=path, version=version)
  return importlib.import_module(mod_name)
