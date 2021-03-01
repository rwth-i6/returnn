"""
This provides the main :func:`import_` function.
"""

import types
import os
import sys
from .git import get_repo_path


_ModuleNamePrefix = "returnn.util.import_._loaded."


def import_(repo, path, version=None):
  """
  :param str repo: e.g. "github.com/rwth-i6/returnn-experiments"
  :param str path: path inside the repo, without starting "/"
  :param str|None version: e.g. "20211231-0123abcd0123". None for development working copy
  :rtype: object|types.ModuleType
  """
  assert path and path[:1] != "/" and ".." not in path
  # This code is only Python >=3.5.
  # noinspection PyUnresolvedReferences,PyCompatibility
  import importlib.util
  repo_path = get_repo_path(repo=repo, version=version)
  full_path = "%s/%s" % (repo_path, path)
  mod_name = _module_name(repo=repo, repo_path=repo_path, path=path, version=version)
  if mod_name in sys.modules:
    return sys.modules[mod_name]
  spec = importlib.util.spec_from_file_location(mod_name, full_path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)  # noqa
  return mod


def _module_name(repo, repo_path, path, version):
  """
  :param str repo:
  :param str repo_path:
  :param str path:
  :param str|None version:
  :rtype: str
  """
  repo_dir_name = os.path.dirname(repo)
  repo_v = "%s/%s" % (repo_dir_name, os.path.basename(repo_path))  # eg "github.com/rwth-i6/returnn-experiments@v..."
  if version:
    repo_v = repo_v.replace("@v", "/v")
  else:
    repo_v = repo_v + "/dev"
  repo_and_path = "%s/%s" % (repo_v, path)
  name = repo_and_path.replace(".", "_").replace("/", ".")
  return _ModuleNamePrefix + name
