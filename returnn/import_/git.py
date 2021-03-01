"""
Provides needed Git utilities to manage the local checked out packages.

The packages are stored in: ``~/returnn/pkg``
(Can be configured via env ``RETURNN_PKG_PATH``)

We might have multiple checkouts of the same repo.
In this case, we want to share the object files.
We are using `git worktree <https://git-scm.com/docs/git-worktree>`__ for this.

References:

  https://github.com/golang/go/blob/724d0720/src/cmd/go/internal/modfetch/codehost/git.go
  https://golang.org/ref/mod

Note about our terminology (esp vs Go terminology):

Go differentiates between packages, modules and repositories,
and even major semantic version sub-directories inside the modules.
We do not make this distinction.
You have a repository, and some specific commit,
and the file path inside the repo.
The file path might be sth like ``transformer-v3.py``
to note that it is incompatible from an earlier version.

Note about our ``version`` (revision) format:

During development, you would want to work in a live working tree.
There is a working tree in ``...repo``, and ``version=None`` refers to that.

Once development of one piece/module/package is ready,
you always should specify the version explicitly.
The format is ``<date>-<commit>``, where ``date`` should be like ``YYYYMMDDHHMMSS``
(can potentially be shorter up to ``YYYYMMDD``)
and ``commit`` is a revision, with minimum 7 numbers of digits.

Note about our ``repo`` format:

Currently always assumed to be remote, and then cached in our repo cache.
"""

import os
import typing
from subprocess import check_call, check_output, SubprocessError
import re

_DefaultPkgPath = "~/returnn/pkg"
_EnvPkgPath = "RETURNN_PKG_PATH"
_MinNumHashDigits = 7
_DefaultNumHashDigits = 12
_FullNumHashDigits = 40
_RevDigitsRe = re.compile("^[0123456789abcdef]*$")
_DateFormat = "%Y%m%d%H%M%S"
_DateFormatLen = 14
_DateFormatAllowedSubLens = (8, 10, 12)


def stat_repo(repo, version):
  """
  :param str repo: e.g. "github.com/rwth-i6/returnn-experiments"
  :param str|None version: e.g. "20211231-0123abcd0123"
  """
  repo_ = _get_repo(repo)
  repo_.stat()
  repo_.get_work_dir(version)


def get_repo_path(repo, version):
  """
  :param str repo: e.g. "github.com/rwth-i6/returnn-experiments"
  :param str|None version: e.g. "20211231-0123abcd0123"
  :return: path to repo
  :rtype: str
  """
  repo_ = _get_repo(repo)
  work_dir = repo_.get_work_dir(version)
  return work_dir.get_path()


def get_repo_file_path(repo, path, version):
  """
  :param str repo: e.g. "github.com/rwth-i6/returnn-experiments"
  :param str path: path inside the repo, without starting "/"
  :param str|None version: e.g. "20211231-0123abcd0123"
  :return: path to local file|dir in our package dir
  :rtype: str

  This also ensures that the returned path exists.
  If the repo is not checked out yet, or not cloned yet,
  it will do so.
  """
  repo_path = get_repo_path(repo, version)
  full_path = "%s/%s" % (repo_path, path)
  assert os.path.exists(full_path)
  return full_path


class _Globals:
  def __init__(self):
    self._package_path = None  # type: typing.Optional[str]

  @property
  def package_path(self):
    """
    :return: directory where packages are stored (default: ~/returnn/pkg)
    :rtype: str
    """
    if self._package_path:
      return self._package_path
    if _EnvPkgPath in os.environ:
      path = os.environ[_EnvPkgPath]
      assert os.path.isdir(path), "import pkg path via env %s: is not a dir: %r" % (_EnvPkgPath, path)
    else:
      path = os.path.expanduser(_DefaultPkgPath)
      os.makedirs(path, exist_ok=True)
    self._package_path = path
    return path


_globals = _Globals()


def _simple_validate_repo_name(repo):
  """
  :param str repo:
  """
  assert ".." not in repo and ":" not in repo


def _main_repo_path(repo):
  """
  :param str repo:
  :return: main repo dir (which includes the Git objects)
  :rtype: str
  """
  return "%s/%s" % (_globals.package_path, repo)


def _dev_repo_path(repo):
  """
  :param str repo:
  :return: dev working tree of a repo. currently the same as the main repo path
  :rtype: str
  """
  return _main_repo_path(repo)


def _repo_path(repo, version):
  """
  :param str repo:
  :param str|None version:
  :rtype: str
  """
  if not version:
    return _dev_repo_path(repo)
  return "%s@v%s" % (_main_repo_path(repo), version)


def _repo_remote_url(repo):
  """
  :param str repo:
  :rtype: str
  """
  # Note: See also Go documentation how it does this: https://golang.org/ref/mod#vcs-find
  # It uses https by default.
  # You can overwrite this via the global Git configuration.
  # Example for GitHub: put this into your ~/.gitconfig:
  # [url "git@github.com:"]
  #     insteadOf = https://github.com/
  p = repo.find("/")
  assert p >= 0
  host, path = repo[:p], repo[p:]
  return "https://%s:%s" % (host, path)


def _simple_validate_commit_rev(rev):
  """
  :param str rev:
  """
  assert len(rev) >= _MinNumHashDigits and _RevDigitsRe.match(rev)


def _rev_from_version(version):
  """
  :param str version: e.g. "20211231-0123abcd0123"
  :return: e.g. "0123abcd0123"
  :rtype: str
  """
  p = version.rfind("-")
  if p < 0:  # ok, allow here
    _simple_validate_commit_rev(version)
    return version
  rev = version[p + 1:]
  _simple_validate_commit_rev(rev)
  return rev


def _version_from_date_and_rev(date, rev):
  """
  :param str date:
  :param str rev:
  """
  _simple_validate_commit_rev(rev)
  return "v%s-%s" % (date, rev)


def _sys_git_clone_repo(repo):
  """
  :param str repo:
  """
  url = _repo_remote_url(repo)
  main_path = _main_repo_path(repo)
  check_call(["git", "clone", url, main_path])


def _sys_git_fetch(repo):
  """
  :param str repo:
  """
  # Note: Just simply git fetch, no extra logic.
  # We might want to allow for named refs later...
  # We might want to extend this logic to be more like Go...
  # https://github.com/golang/go/blob/724d0720b3e/src/cmd/go/internal/modfetch/codehost/git.go#L366
  main_path = _main_repo_path(repo)
  check_call(["git", "fetch"], cwd=main_path)


def _sys_git_create_repo_workdir(repo, version):
  """
  :param str repo:
  :param str version:
  """
  main_path = _main_repo_path(repo)
  rev = _rev_from_version(version)
  version_path = _repo_path(repo, version)
  check_call(
    ["git", "worktree", "add", version_path, rev],
    cwd=main_path)


def _sys_git_stat_local_rev(repo, rev):
  """
  :param str repo:
  :param str rev:
  :return: (full_rev, date)
  :rtype: (str|None, str|None)
  """
  main_path = _main_repo_path(repo)
  try:
    out = check_output(
      ["git",
       "-c", "log.showsignature=false",
       "log", "-n1",
       "--format=format:%H %cd",
       "--date=format:%s" % _DateFormat,
       rev, "--"],
      cwd=main_path)
  except SubprocessError:
    return None, None
  out = out.decode("utf8")
  full_rev, date = out.split()
  assert full_rev.startswith(rev) and len(full_rev) == _FullNumHashDigits
  return full_rev, date


class _Repo:
  def __init__(self, name):
    """
    :param str name: e.g. "github.com/rwth-i6/returnn-experiments"
    """
    _simple_validate_repo_name(name)
    self.name = name
    self._cloned = False
    self._loaded_local_work_dirs = False
    self._dev_work_dir = None  # type: typing.Optional[_RepoWorkDir]
    self._work_dirs = {}  # type: typing.Dict[str,_RepoWorkDir]  # commit-rev -> entry

  def stat(self):
    """
    Stat repo.
    """
    self._clone()

  def get_dev_dir_path(self):
    """
    :rtype: str
    """
    self._clone()
    return _dev_repo_path(self.name)

  def _clone(self):
    if self._cloned:
      return
    main_path = _main_repo_path(self.name)
    if not os.path.exists(main_path):
      _sys_git_clone_repo(self.name)
      assert os.path.exists(main_path)
    self._cloned = True

  def _load_work_dirs(self):
    if self._loaded_local_work_dirs:
      return
    self._clone()
    main_path = _main_repo_path(self.name)
    p = main_path.rfind("/")
    assert p > 0
    dir_name, base_name = main_path[:p], main_path[p + 1:]
    prefix = base_name + "@v"
    for d in sorted(os.listdir(dir_name)):
      if not d.startswith(prefix):
        continue
      version = d[len(prefix):]
      rev = _rev_from_version(version)
      if rev in self._work_dirs:
        continue
      self._work_dirs[rev] = _RepoWorkDir(repo=self, version=version)
    self._loaded_local_work_dirs = True

  def _get_work_dir_stat_local(self, rev):
    """
    :param str rev:
    :rtype: _RepoWorkDir
    """
    full_rev, date = _sys_git_stat_local_rev(self.name, rev)
    if full_rev:
      default_rev = full_rev[:_DefaultNumHashDigits]
      assert default_rev not in self._work_dirs
      default_version = _version_from_date_and_rev(date=date, rev=default_rev)
      _sys_git_create_repo_workdir(self.name, version=default_version)
      work_dir = _RepoWorkDir(self, default_version)
      self._work_dirs[default_rev] = work_dir
      return work_dir
    return None

  def get_work_dir(self, version):
    """
    :param str|None version:
    :rtype: _RepoWorkDir
    """
    if not version:
      if self._dev_work_dir:
        return self._dev_work_dir
      self._dev_work_dir = _RepoWorkDir(self, None)
      return self._dev_work_dir
    self._load_work_dirs()
    rev = _rev_from_version(version)
    if rev in self._work_dirs:
      return self._work_dirs[rev]
    for rev_, work_dir in self._work_dirs.items():
      if rev_.startswith(rev) or (len(rev) > len(rev_) and rev.startswith(rev_)):
        self._work_dirs[rev] = work_dir
        return work_dir
    work_dir = self._get_work_dir_stat_local(rev)
    if work_dir:
      return work_dir
    _sys_git_fetch(self.name)
    work_dir = self._get_work_dir_stat_local(rev)
    assert work_dir, "Git repo %s, version %s unknown." % (self.name, version)
    return work_dir


class _RepoWorkDir:
  def __init__(self, repo, version):
    """
    :param str|_Repo repo:
    :param str|None version: (normalized)
    """
    if not isinstance(repo, _Repo):
      repo = _get_repo(repo)
    self.repo = repo
    self.version = version

  def get_path(self):
    """
    :rtype: str
    """
    return _repo_path(self.repo.name, self.version)


_repo_cache = {}  # type: typing.Dict[str,_Repo]


def _get_repo(repo):
  """
  :param str repo:
  :rtype: _Repo
  """
  assert isinstance(repo, str)
  obj = _repo_cache.get(repo)
  if obj:
    return obj
  obj = _Repo(repo)
  _repo_cache[repo] = obj
  return obj
