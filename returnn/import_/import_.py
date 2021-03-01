"""
This provides the main :func:`import_` function.
"""

import types
import os
import importlib
from .git import get_repo_path


_MyDir = os.path.dirname(os.path.abspath(__file__))
_PyPkgSymlinkDir = _MyDir + "/_pkg"
_ModuleNamePrefix = "returnn.import_._pkg."


def import_(repo, path, version=None):
  """
  :param str repo: e.g. "github.com/rwth-i6/returnn-experiments"
  :param str path: path inside the repo, without starting "/"
  :param str|None version: e.g. "20211231-0123abcd0123". None for development working copy
  :rtype: object|types.ModuleType
  """
  assert path and path[:1] != "/" and ".." not in path
  repo_path = get_repo_path(repo=repo, version=version)
  mod_name = _module_name(repo=repo, repo_path=repo_path, path=path, version=version)
  return importlib.import_module(mod_name)


def _module_name(repo, repo_path, path, version):
  """
  :param str repo:
  :param str repo_path:
  :param str path:
  :param str|None version:
  :rtype: str
  """
  full_path = "%s/%s" % (repo_path, path)
  py_pkg_dirname = _find_root_python_package(full_path)
  assert len(py_pkg_dirname) >= len(repo_path)
  rel_pkg_path = full_path[len(py_pkg_dirname) + 1:]
  p = rel_pkg_path.find("/")
  if p > 0:
    rel_pkg_path0 = rel_pkg_path[:p]
  else:
    rel_pkg_path0 = rel_pkg_path
  rel_pkg_dir = py_pkg_dirname[len(repo_path):]  # starting with "/"

  repo_dir_name = os.path.dirname(repo)
  repo_v = "%s/%s" % (repo_dir_name, os.path.basename(repo_path))  # eg "github.com/rwth-i6/returnn-experiments@v..."
  if version:
    repo_v = repo_v.replace("@v", "/v")
  else:
    repo_v = repo_v + "/dev"
  py_pkg_dir = "%s/%s%s" % (_PyPkgSymlinkDir, repo_v, rel_pkg_dir)
  _mk_py_pkg_dirs(_PyPkgSymlinkDir, py_pkg_dir)
  symlink_file = "%s/%s" % (py_pkg_dir, rel_pkg_path0)
  symlink_target = "%s%s/%s" % (repo_path, rel_pkg_dir, rel_pkg_path0)
  if os.path.exists(symlink_file):
    assert os.readlink(symlink_file) == symlink_target
  else:
    os.symlink(symlink_target, symlink_file, target_is_directory=os.path.isdir(symlink_target))

  repo_and_path = "%s/%s" % (repo_v, path[:-3] if path.endswith(".py") else path)
  name = repo_and_path.replace(".", "_").replace("/", ".")
  return _ModuleNamePrefix + name


def _mk_py_pkg_dirs(path, path_=None):
  """
  :param str path:
  :param str|None path_:
  """
  if path_:
    assert path_.startswith(path + "/")
  while True:
    if os.path.exists(path):
      assert os.path.isdir(path) and not os.path.islink(path) and os.path.exists(path + "/__init__.py")
    else:
      os.mkdir(path)
      open(path + "/__init__.py", "x").close()
    if path_:
      if len(path) == len(path_):
        break
      p = path_.find("/", len(path) + 1)
      if p > 0:
        path = path_[:p]
      else:
        break
    else:
      break


def _find_root_python_package(full_path):
  """
  :param str full_path: some Python file
  :return: going up from path, and first dir which does not include __init__.py
  :rtype: str
  """
  p = len(full_path)
  while True:
    p = full_path.rfind("/", 0, p)
    assert p > 0
    d = full_path[:p]
    if not os.path.exists(d + "/__init__.py"):
      return d
