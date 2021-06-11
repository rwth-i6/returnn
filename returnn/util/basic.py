# -*- coding: utf8 -*-

"""
Various generic utilities, which are shared across different backend engines.
"""

from __future__ import print_function
from __future__ import division

import subprocess
from subprocess import CalledProcessError

import h5py
from collections import deque
import inspect
import os
import sys
import shlex
import numpy as np
import re
import time
import contextlib
try:
  import thread
except ImportError:
  import _thread as thread
import threading
try:
  # noinspection PyCompatibility
  from StringIO import StringIO
except ImportError:
  from io import StringIO
import typing
from returnn.log import log

PY3 = sys.version_info[0] >= 3

if PY3:
  import builtins
  unicode = str
  long = int
  # noinspection PyShadowingBuiltins
  input = builtins.input
  from io import BytesIO
else:
  # noinspection PyUnresolvedReferences
  import __builtin__ as builtins
  unicode = builtins.unicode  # type: typing.Type[str]
  long = builtins.long  # type: typing.Type[int]
  # noinspection PyShadowingBuiltins
  input = builtins.raw_input
  try:
    # noinspection PyUnresolvedReferences,PyCompatibility
    from cStringIO import StringIO as BytesIO
  except ImportError:
    # noinspection PyUnresolvedReferences,PyCompatibility
    from StringIO import StringIO as BytesIO

returnn_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class NotSpecified(object):
  """
  This is just a placeholder, to be used as default argument to mark that it is not specified.
  """
  def __str__(self):
    return "NotSpecified"

  def __repr__(self):
    return "<NotSpecified>"

  @classmethod
  def resolve(cls, value, default):
    """
    :param T|NotSpecified|type[NotSpecified] value:
    :param U default:
    :rtype: T|U
    """
    if value is NotSpecified:
      return default
    return value


class OptionalNotImplementedError(NotImplementedError):
  """
  This can optionally be implemented, but it is not required by the API.
  """


def is_64bit_platform():
  """
  :return: True if we run on 64bit, False for 32bit
  :rtype: bool
  http://stackoverflow.com/questions/1405913/how-do-i-determine-if-my-python-shell-is-executing-in-32bit-or-64bit-mode-on-os
  """
  return sys.maxsize > 2**32


class BackendEngine:
  """
  Stores which backend engine we use in RETURNN.
  E.g. Theano or TensorFlow.
  """

  Theano = 0
  TensorFlow = 1
  selectedEngine = None  # type: typing.Optional[int]  # One of the possible engines.

  class CannotSelectEngine(Exception):
    """
    Cannot select backend engine
    """

  @classmethod
  def select_engine(cls, engine=None, config=None):
    """
    :param int engine: see the global class attribs for possible values
    :param Config.Config config:
    """
    assert cls.selectedEngine is None, "already set"
    if engine is None:
      if config is None:
        from returnn.config import get_global_config
        config = get_global_config()
      engine = cls._get_default_engine()
      if config.bool("use_theano", False):
        engine = cls.Theano
      if config.bool("use_tensorflow", False):
        engine = cls.TensorFlow
    cls.selectedEngine = engine

  @classmethod
  def _get_default_engine(cls):
    """
    For backward compatibility, we still keep Theano the default.
    However, we can do it slightly more clever.
    If TensorFlow is already imported but Theano is not, allow TF as the default.
    Also, if theano cannot be imported, allow for TF as the default.

    :rtype: int
    """
    if "theano" in sys.modules:
      return cls.Theano
    if "tensorflow" in sys.modules:
      return cls.TensorFlow
    try:
      # noinspection PyPackageRequirements
      import theano
      return cls.Theano
    except ImportError:
      pass
    try:
      import tensorflow
      return cls.TensorFlow
    except ImportError:
      pass
    raise cls.CannotSelectEngine("Neither Theano nor TF available.")

  @classmethod
  def get_selected_engine(cls):
    """
    :rtype: int
    """
    if cls.selectedEngine is not None:
      return cls.selectedEngine
    else:
      return cls._get_default_engine()

  @classmethod
  def is_theano_selected(cls):
    """
    :rtype: bool
    """
    return cls.get_selected_engine() == cls.Theano

  @classmethod
  def is_tensorflow_selected(cls):
    """
    :rtype: bool
    """
    return cls.get_selected_engine() == cls.TensorFlow


class BehaviorVersion:
  """
  Stores the global behavior_version

  The version will be set after the config is defined at __main__.init_config() or Engine.__init__()
  """

  _behavior_version = 0  # type: typing.Optional[int]
  _is_set = False  # type: bool

  @classmethod
  def set(cls, version):
    """
    :param int version:
    """
    assert not cls._is_set, "behavior version already set"
    assert version >= 0
    cls._behavior_version = version
    cls._is_set = True

  @classmethod
  def get(cls):
    """
    :rtype: int
    """
    return cls._behavior_version

  @classmethod
  def is_set(cls):
    """
    :rtype: bool
    """
    return cls._is_set

  @classmethod
  def require(cls, condition, message, version):
    """
    :param bool condition:
    :param str message:
    :param int version:
    """
    if not condition:
      if BehaviorVersion.get() >= version:
        raise Exception(message)
      else:
        log.print_deprecation_warning(message, behavior_version=version)


def get_model_filename_postfix():
  """
  :return: one possible postfix of a file which will be present when the model is saved
  :rtype: str
  """
  assert BackendEngine.selectedEngine is not None
  if BackendEngine.is_tensorflow_selected():
    # There will be multiple files but a *.meta file will always be present.
    return ".meta"
  return ""


def sys_cmd_out_lines(s):
  """
  :param str s: shell command
  :rtype: list[str]
  :return: all stdout split by newline. Does not cover stderr.
  Raises CalledProcessError on error.
  """
  p = subprocess.Popen(s, stdout=subprocess.PIPE, shell=True, close_fds=True,
                       env=dict(os.environ, LANG="en_US.UTF-8", LC_ALL="en_US.UTF-8"))
  stdout = p.communicate()[0]
  if PY3:
    stdout = stdout.decode("utf8")
  result = [line.strip() for line in stdout.split('\n')[:-1]]
  p.stdout.close()
  if p.returncode != 0:
    raise CalledProcessError(p.returncode, s, stdout)
  return result


def sys_exec_out(*args, **kwargs):
  """
  :param str args: for subprocess.Popen
  :param kwargs: for subprocess.Popen
  :return: stdout as str (assumes utf8)
  :rtype: str
  """
  from subprocess import Popen, PIPE
  kwargs.setdefault("shell", False)
  p = Popen(args, stdin=PIPE, stdout=PIPE, **kwargs)
  out, _ = p.communicate()
  if p.returncode != 0:
    raise CalledProcessError(p.returncode, args)
  out = unicode_to_str(out)
  return out


def sys_exec_ret_code(*args, **kwargs):
  """
  :param str args: for subprocess.call
  :param kwargs: for subprocess.call
  :return: return code
  :rtype: int
  """
  import subprocess
  res = subprocess.call(args, shell=False, **kwargs)
  valid = kwargs.get("valid", (0, 1))
  if valid is not None:
    if res not in valid:
      raise CalledProcessError(res, args)
  return res


def git_commit_rev(commit="HEAD", git_dir=".", length=None):
  """
  :param str commit:
  :param str git_dir:
  :param int|None length:
  :rtype: str
  """
  if commit is None:
    commit = "HEAD"
  return sys_exec_out(
    "git", "rev-parse", "--short=%i" % length if length else "--short", commit,
    cwd=git_dir).strip()


def git_is_dirty(git_dir="."):
  """
  :param str git_dir:
  :rtype: bool
  """
  r = sys_exec_ret_code("git", "diff", "--no-ext-diff", "--quiet", "--exit-code", cwd=git_dir)
  if r == 0:
    return False
  if r == 1:
    return True
  assert False, "bad return %i" % r


def git_commit_date(commit="HEAD", git_dir="."):
  """
  :param str commit:
  :param str git_dir:
  :rtype: str
  """
  return (
    sys_exec_out("git", "show", "-s", "--format=%ci", commit, cwd=git_dir)
    .strip()[:-6]
    .replace(":", "")
    .replace("-", "")
    .replace(" ", "."))


def git_describe_head_version(git_dir="."):
  """
  :param str git_dir:
  :rtype: str
  """
  cdate = git_commit_date(git_dir=git_dir)
  rev = git_commit_rev(git_dir=git_dir)
  is_dirty = git_is_dirty(git_dir=git_dir)
  return "%s--git-%s%s" % (cdate, rev, "-dirty" if is_dirty else "")


def describe_returnn_version():
  """
  :rtype: str
  :return: string like "1.20171017.163840+git-ab2a1da"
  """
  from returnn import __long_version__
  return __long_version__


def describe_theano_version():
  """
  :rtype: str
  """
  # noinspection PyUnresolvedReferences,PyPackageRequirements
  import theano
  try:
    theano_dir = os.path.dirname(theano.__file__)
  except Exception as e:
    theano_dir = "<unknown(exception: %r)>" % e
  try:
    version = theano.__version__
    if len(version) > 20:
      version = version[:20] + "..."
  except Exception as e:
    version = "<unknown(exception: %r)>" % e
  try:
    if theano_dir.startswith("<"):
      git_info = "<unknown-dir>"
    elif os.path.exists(theano_dir + "/../.git"):
      git_info = "git:" + git_describe_head_version(git_dir=theano_dir)
    elif "/site-packages/" in theano_dir:
      git_info = "<site-package>"
    else:
      git_info = "<not-under-git>"
  except Exception as e:
    git_info = "<unknown(git exception: %r)>" % e
  return "%s (%s in %s)" % (version, git_info, theano_dir)


def describe_tensorflow_version():
  """
  :rtype: str
  """
  try:
    import tensorflow as tf
  except ImportError:
    return "<TensorFlow ImportError>"
  try:
    tdir = os.path.dirname(tf.__file__)
  except Exception as e:
    tdir = "<unknown(exception: %r)>" % e
  version = getattr(tf, "__version__", "<unknown version>")
  version += " (%s)" % getattr(tf, "__git_version__", "<unknown git version>")
  try:
    if tdir.startswith("<"):
      git_info = "<unknown-dir>"
    elif os.path.exists(tdir + "/../.git"):
      git_info = "git:" + git_describe_head_version(git_dir=tdir)
    elif "/site-packages/" in tdir:
      git_info = "<site-package>"
    else:
      git_info = "<not-under-git>"
  except Exception as e:
    git_info = "<unknown(git exception: %r)>" % e
  return "%s (%s in %s)" % (version, git_info, tdir)


def get_tensorflow_version_tuple():
  """
  :return: tuple of ints, first entry is the major version
  :rtype: tuple[int]
  """
  import tensorflow as tf
  import re
  return tuple([int(re.sub('(-rc[0-9]|-dev[0-9]*)', '', s)) for s in tf.__version__.split(".")])


def eval_shell_env(token):
  """
  :param str token:
  :return: if "$var", looks in os.environ, otherwise return token as is
  :rtype: str
  """
  if token.startswith("$"):
    return os.environ.get(token[1:], "")
  return token


def eval_shell_str(s):
  """
  :type s: str | list[str] | ()->str | list[()->str] | ()->list[str] | ()->list[()->str]
  :rtype: list[str]

  Parses `s` as shell like arguments (via shlex.split) and evaluates shell environment variables
  (:func:`eval_shell_env`).
  `s` or its elements can also be callable. In those cases, they will be called and the returned value is used.
  """
  tokens = []
  if callable(s):
    s = s()
  if isinstance(s, (list, tuple)):
    ls = s
  else:
    assert isinstance(s, (str, unicode))
    ls = shlex.split(s)
  for token in ls:
    if callable(token):
      token = token()
    assert isinstance(token, (str, unicode))
    if token.startswith("$"):
      tokens += eval_shell_str(eval_shell_env(token))
    else:
      tokens += [token]
  return tokens


def hdf5_dimension(filename, dimension):
  """
  :param str filename:
  :param str dimension:
  :rtype: numpy.ndarray|int
  """
  fin = h5py.File(filename, "r")
  if '/' in dimension:
    res = fin['/'.join(dimension.split('/')[:-1])].attrs[dimension.split('/')[-1]]
  else:
    res = fin.attrs[dimension]
  fin.close()
  return res


def hdf5_group(filename, dimension):
  """
  :param str filename:
  :param str dimension:
  :rtype: dict[str]
  """
  fin = h5py.File(filename, "r")
  res = {k: fin[dimension].attrs[k] for k in fin[dimension].attrs}
  fin.close()
  return res


def hdf5_shape(filename, dimension):
  """
  :param str filename:
  :param dimension:
  :rtype: tuple[int]
  """
  fin = h5py.File(filename, "r")
  res = fin[dimension].shape
  fin.close()
  return res


def hdf5_strings(handle, name, data):
  """
  :param h5py.File handle:
  :param str name:
  :param numpy.ndarray|list[str] data:
  """
  # noinspection PyBroadException
  try:
    s = max([len(d) for d in data])
    dset = handle.create_dataset(name, (len(data),), dtype="S" + str(s))
    dset[...] = data
  except Exception:
    # noinspection PyUnresolvedReferences
    dt = h5py.special_dtype(vlen=unicode)
    del handle[name]
    dset = handle.create_dataset(name, (len(data),), dtype=dt)
    dset[...] = data


def model_epoch_from_filename(filename):
  """
  :param str filename:
  :return: epoch number
  :rtype: int
  """
  if BackendEngine.is_theano_selected():
    return hdf5_dimension(filename, 'epoch')
  else:
    # We could check via:
    # tf.contrib.framework.python.framework.checkpoint_utils.load_variable()
    # once we save that in the model.
    # See TFNetwork.Network._create_saver().
    # We don't have it in the model, though.
    # For now, just parse it from filename.
    # If TF, and symlink, resolve until no symlink anymore (e.g. if we symlinked the best epoch).
    while True:
      tf_meta_fn = "%s.meta" % filename
      if os.path.exists(tf_meta_fn) and os.path.islink(tf_meta_fn):
        tf_meta_fn_ = os.readlink(tf_meta_fn)
        assert tf_meta_fn_.endswith(".meta"), "strange? %s, %s" % (filename, tf_meta_fn)
        filename = tf_meta_fn_[:-len(".meta")]
      else:
        break
    m = re.match(".*\\.([0-9]+)", filename)
    assert m, "no match for %r" % filename
    return int(m.groups()[0])


def deep_update_dict_values(d, key, new_value):
  """
  Visits all items in `d`.
  If the value is a dict, it will recursively visit it.

  :param dict[str,T|object|None|dict] d: will update inplace
  :param str key:
  :param T new_value:
  """
  for value in d.values():
    if isinstance(value, dict):
      deep_update_dict_values(value, key=key, new_value=new_value)
  if key in d:
    d[key] = new_value


def terminal_size(file=sys.stdout):
  """
  Returns the terminal size.
  This will probably work on linux only.

  :param io.File file:
  :return: (columns, lines), or (-1,-1)
  :rtype: (int,int)
  """
  import os
  import io
  if not hasattr(file, "fileno"):
    return -1, -1
  try:
    if not os.isatty(file.fileno()):
      return -1, -1
  except io.UnsupportedOperation:
    return -1, -1
  env = os.environ

  # noinspection PyShadowingNames
  def ioctl_gwinsz(fd):
    """
    :param int fd: file descriptor
    :rtype: tuple[int]
    """
    # noinspection PyBroadException
    try:
      import fcntl
      import termios
      import struct
      cr_ = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))  # noqa
    except Exception:
      return
    return cr_

  cr = ioctl_gwinsz(file.fileno) or ioctl_gwinsz(0) or ioctl_gwinsz(1) or ioctl_gwinsz(2)
  if not cr:
    # noinspection PyBroadException
    try:
      fd = os.open(os.ctermid(), os.O_RDONLY)
      cr = ioctl_gwinsz(fd)
      os.close(fd)
    except Exception:
      pass
  if not cr:
    cr = (env.get('LINES', 25), env.get('COLUMNS', 80))
  return int(cr[1]), int(cr[0])


def is_tty(file=sys.stdout):
  """
  :param io.File file:
  :rtype: bool
  """
  terminal_width, _ = terminal_size(file=file)
  return terminal_width > 0


def confirm(txt, exit_on_false=False):
  """
  :param str txt: e.g. "Delete everything?"
  :param bool exit_on_false: if True, will call sys.exit(1) if not confirmed
  :rtype: bool
  """
  while True:
    r = input("%s Confirm? [yes/no]" % txt)
    if not r:
      continue
    if r in ["y", "yes"]:
      return True
    if r in ["n", "no"]:
      if exit_on_false:
        sys.exit(1)
      return False
    print("Invalid response %r." % r)


def hms(s):
  """
  :param float|int s: seconds
  :return: e.g. "1:23:45" (hs:ms:secs). see hms_fraction if you want to get fractional seconds
  :rtype: str
  """
  m, s = divmod(s, 60)
  h, m = divmod(m, 60)
  return "%d:%02d:%02d" % (h, m, s)


def hms_fraction(s, decimals=4):
  """
  :param float s: seconds
  :param int decimals: how much decimals to print
  :return: e.g. "1:23:45.6789" (hs:ms:secs)
  :rtype: str
  """
  return hms(int(s)) + (("%%.0%if" % decimals) % (s - int(s)))[1:]


def human_size(n, factor=1000, frac=0.8, prec=1):
  """
  :param int|float n:
  :param int factor: for each of the units K, M, G, T
  :param float frac: when to go over to the next bigger unit
  :param int prec: how much decimals after the dot
  :return: human readable size, using K, M, G, T
  :rtype: str
  """
  postfixes = ["", "K", "M", "G", "T"]
  i = 0
  while i < len(postfixes) - 1 and n > (factor ** (i + 1)) * frac:
    i += 1
  if i == 0:
    return str(n)
  return ("%." + str(prec) + "f") % (float(n) / (factor ** i)) + postfixes[i]


def human_bytes_size(n, factor=1024, frac=0.8, prec=1):
  """
  :param int|float n:
  :param int factor: see :func:`human_size`. 1024 by default for bytes
  :param float frac: see :func:`human_size`
  :param int prec: how much decimals after the dot
  :return: human readable byte size, using K, M, G, T, with "B" at the end
  :rtype: str
  """
  return human_size(n, factor=factor, frac=frac, prec=prec) + "B"


def _pp_extra_info(obj, depth_limit=3):
  """
  :param object obj:
  :param int depth_limit:
  :return: extra info (if available: len, some items, ...)
  :rtype: str
  """
  if isinstance(obj, np.ndarray):
    return "shape=%r" % (obj.shape,)
  s = []
  if hasattr(obj, "__len__"):
    # noinspection PyBroadException
    try:
      # noinspection PyTypeChecker
      if type(obj) in (str, unicode, list, tuple, dict) and len(obj) <= 5:
        pass  # don't print len in this case
      else:
        s += ["len = %i" % obj.__len__()]  # noqa
    except Exception:
      pass
  if depth_limit > 0 and hasattr(obj, "__getitem__"):
    # noinspection PyBroadException
    try:
      if type(obj) in (str, unicode):
        pass  # doesn't make sense to get sub items here
      else:
        sub_obj = obj.__getitem__(0)  # noqa
        extra_info = _pp_extra_info(sub_obj, depth_limit - 1)
        if extra_info != "":
          s += ["_[0]: {%s}" % extra_info]
    except Exception:
      pass
  return ", ".join(s)


_pretty_print_limit = 300
_pretty_print_as_bytes = False


def set_pretty_print_default_limit(limit):
  """
  :param int|float limit: use float("inf") to disable
  """
  global _pretty_print_limit
  _pretty_print_limit = limit


def set_pretty_print_as_bytes(as_bytes):
  """
  :param bool as_bytes:
  """
  global _pretty_print_as_bytes
  _pretty_print_as_bytes = as_bytes


def pretty_print(obj, limit=None):
  """
  :param object obj:
  :param int|float limit: use float("inf") to disable. None will use the default, via set_pretty_print_default_limit
  :return: repr(obj), or some shorted version of that, maybe with extra info
  :rtype: str
  """
  if _pretty_print_as_bytes and isinstance(obj, np.ndarray):
    bs = obj.tobytes()
    import gzip
    bs = gzip.compress(bs)
    import base64
    if len(bs) > 57:
      parts = []
      while len(bs) > 0:
        parts.append(bs[:57])
        bs = bs[57:]
        if len(bs) == 0:
          break
      s = "\n  " + "\n  ".join([repr(base64.encodebytes(bs).strip()) for bs in parts]) + "\n  "
    else:
      s = repr(base64.encodebytes(bs).strip())
    s = "numpy.frombuffer(gzip.decompress(base64.decodebytes(%s)), dtype=%r).reshape(%r)" % (
      s, str(obj.dtype), obj.shape)
  else:
    s = repr(obj)
  if limit is None:
    limit = _pretty_print_limit
  if len(s) > limit:
    s = s[:int(limit) - 3]
    s += "..."
  extra_info = _pp_extra_info(obj)
  if extra_info != "":
    s += ", " + extra_info
  return s


def progress_bar(complete=1.0, prefix="", suffix="", file=None):
  """
  Prints some progress bar.

  :param float complete: from 0.0 to 1.0
  :param str prefix:
  :param str suffix:
  :param io.TextIOWrapper|typing.TextIO|None file: where to print. stdout by default
  :return: nothing, will print on ``file``
  """
  if file is None:
    file = sys.stdout
  terminal_width, _ = terminal_size(file=file)
  if terminal_width == -1:
    return
  if complete == 1.0:
    file.write("\r%s" % (terminal_width * ' '))
    file.flush()
    file.write("\r")
    file.flush()
    return
  progress = "%.02f%%" % (complete * 100)
  if prefix != "":
    prefix = prefix + " "
  if suffix != "":
    suffix = " " + suffix
  ntotal = terminal_width - len(progress) - len(prefix) - len(suffix) - 4
  bars = '|' * int(complete * ntotal)
  spaces = ' ' * (ntotal - int(complete * ntotal))
  bar = bars + spaces
  file.write("\r%s" % prefix + "[" + bar[:len(bar)//2] + " " + progress + " " + bar[len(bar)//2:] + "]" + suffix)
  file.flush()


class _ProgressBarWithTimeStats:
  """
  Global closure. Used by :func:`progress_bar_with_time`.
  """
  start_time = None
  last_complete = None


def progress_bar_with_time(complete=1.0, prefix="", **kwargs):
  """
  :func:`progress_bar` with additional remaining time estimation.

  :param float complete:
  :param str prefix:
  :param kwargs: passed to :func:`progress_bar`
  :return: nothing
  """
  stats = _ProgressBarWithTimeStats
  if stats.start_time is None:
    stats.start_time = time.time()
    stats.last_complete = complete
  if stats.last_complete > complete:
    stats.start_time = time.time()
  stats.last_complete = complete

  start_elapsed = time.time() - stats.start_time
  if complete > 0:
    total_time_estimated = start_elapsed / complete
    remaining_estimated = total_time_estimated - start_elapsed
    if prefix:
      prefix += ", " + hms(remaining_estimated)
    else:
      prefix = hms(remaining_estimated)
  progress_bar(complete, prefix=prefix, **kwargs)


def available_physical_memory_in_bytes():
  """
  :rtype: int
  """
  # noinspection PyBroadException
  try:
    mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
  except Exception:
    mem_bytes = 1024 ** 4  # just some random number, 1TB
  return mem_bytes


def default_cache_size_in_gbytes(factor=0.7):
  """
  :param float|int factor:
  :rtype: int
  """
  mem_gbytes = available_physical_memory_in_bytes() / (1024. ** 3)
  return int(mem_gbytes * factor)


def better_repr(o):
  """
  The main difference to :func:`repr`: this one is deterministic.
  The orig dict.__repr__ has the order undefined for dict or set.
  For big dicts/sets/lists, add "," at the end to make textual diffs nicer.

  :param object o:
  :rtype: str
  """
  if isinstance(o, list):
    return "[\n%s]" % "".join(map(lambda v: better_repr(v) + ",\n", o))
  if isinstance(o, deque):
    return "deque([\n%s])" % "".join(map(lambda v: better_repr(v) + ",\n", o))
  if isinstance(o, tuple):
    if len(o) == 1:
      return "(%s,)" % o[0]
    return "(%s)" % ", ".join(map(better_repr, o))
  if isinstance(o, dict):
    ls = [better_repr(k) + ": " + better_repr(v) for (k, v) in sorted(o.items())]
    if sum([len(v) for v in ls]) >= 40:
      return "{\n%s}" % "".join([v + ",\n" for v in ls])
    else:
      return "{%s}" % ", ".join(ls)
  if isinstance(o, set):
    return "set([\n%s])" % "".join(map(lambda v: better_repr(v) + ",\n", o))
  # fallback
  return repr(o)


def simple_obj_repr(obj):
  """
  :return: All self.__init__ args.
  :rtype: str
  """
  return obj.__class__.__name__ + "(%s)" % \
                                  ", ".join(["%s=%s" % (arg, better_repr(getattr(obj, arg)))
                                             for arg in getargspec(obj.__init__).args[1:]])


class ObjAsDict(typing.Mapping[str, object]):
  """
  Wraps up any object as a dict, where the attributes becomes the keys.
  See also :class:`DictAsObj`.
  """

  def __init__(self, obj):
    self.__obj = obj

  def __getitem__(self, item):
    if not isinstance(item, (str, unicode)):
      raise KeyError(item)
    try:
      return getattr(self.__obj, item)
    except AttributeError as e:
      raise KeyError(e)

  def __len__(self):
    return len(vars(self.__obj))

  def __iter__(self):
    return iter(vars(self.__obj))

  def items(self):
    """
    :return: vars(..).items()
    :rtype: set[(str,object)]
    """
    return vars(self.__obj).items()


class DictAsObj:
  """
  Wraps up any dictionary as an object, where the keys becomes the attributes.
  See also :class:`ObjAsDict`.
  """

  def __init__(self, dikt):
    """
    :param dict[str] dikt:
    """
    self.__dict__ = dikt


def dict_joined(*ds):
  """
  :param dict[T,V] ds:
  :return: all dicts joined together
  :rtype: dict[T,V]
  """
  res = {}
  for d in ds:
    res.update(d)
  return res


def obj_diff_str(self, other):
  """
  :param object self:
  :param object other:
  :return: the difference described
  :rtype: str
  """
  if self is None and other is None:
    return "No diff."
  if self is None and other is not None:
    return "self is None and other is %r" % other
  if self is not None and other is None:
    return "other is None and self is %r" % self
  if self == other:
    return "No diff."
  s = []

  def _obj_attribs(obj):
    """
    :param object obj:
    :rtype: list[str]|None
    """
    d = getattr(obj, "__dict__", None)
    if d is not None:
      return d.keys()
    return None

  class _NotSpecified:
    def __repr__(self):
      return "<not-specified>"
  not_specified = _NotSpecified()

  self_attribs = _obj_attribs(self)
  other_attribs = _obj_attribs(other)
  if self_attribs is None or other_attribs is None:
    return "self: %r, other: %r" % (self, other)
  for attrib in sorted(set(self_attribs).union(other_attribs)):
    value_self = getattr(self, attrib, not_specified)
    value_other = getattr(other, attrib, not_specified)
    if isinstance(value_self, list):
      if not isinstance(value_other, list):
        s += ["attrib %r self is list but other is %r" % (attrib, type(value_other))]
      elif len(value_self) != len(value_other):
        s += ["attrib %r list differ. len self: %i, len other: %i" % (attrib, len(value_self), len(value_other))]
      else:
        for i, (a, b) in enumerate(zip(value_self, value_other)):
          if a != b:
            s += ["attrib %r[%i] differ. self: %r, other: %r" % (attrib, i, a, b)]
    elif isinstance(value_self, dict):
      if not isinstance(value_other, dict):
        s += ["attrib %r self is dict but other is %r" % (attrib, type(value_other))]
      elif value_self != value_other:
        s += ["attrib %r dict differs:" % attrib]
        s += ["  " + line for line in dict_diff_str(value_self, value_other).splitlines()]
    else:
      if value_self != value_other:
        s += ["attrib %r differ. self: %r, other: %r" % (attrib, value_self, value_other)]
  if s:
    return "\n".join(s)
  else:
    return "No diff."


def dict_diff_str(self, other):
  """
  :param dict self:
  :param dict other:
  :return: the difference described
  :rtype: str
  """
  return obj_diff_str(DictAsObj(self), DictAsObj(other))


def find_ranges(ls):
  """
  :type ls: list[int]
  :returns list of ranges (start,end) where end is exclusive
  such that the union of range(start,end) matches l.
  :rtype: list[(int,int)]
  We expect that the incoming list is sorted and strongly monotonic increasing.
  """
  if not ls:
    return []
  ranges = [(ls[0], ls[0])]
  for k in ls:
    assert k >= ranges[-1][1]  # strongly monotonic increasing
    if k == ranges[-1][1]:
      ranges[-1] = (ranges[-1][0], k + 1)
    else:
      ranges += [(k, k + 1)]
  return ranges


_thread_join_hack_installed = False


def init_thread_join_hack():
  """
  ``threading.Thread.join`` and ``threading.Condition.wait`` would block signals when run in the main thread.
  We never want to block signals.
  Here we patch away that behavior.
  """
  global _thread_join_hack_installed
  if _thread_join_hack_installed:  # don't install twice
    return
  _thread_join_hack_installed = True
  main_thread = threading.currentThread()
  # noinspection PyUnresolvedReferences,PyProtectedMember
  assert isinstance(main_thread, threading._MainThread)
  main_thread_id = thread.get_ident()

  # Patch Thread.join().
  join_orig = threading.Thread.join

  def join_hacked(thread_obj, timeout=None):
    """
    :type thread_obj: threading.Thread
    :type timeout: float|None
    :return: always None
    """
    if thread.get_ident() == main_thread_id and timeout is None:
      # This is a HACK for Thread.join() if we are in the main thread.
      # In that case, a Thread.join(timeout=None) would hang and even not respond to signals
      # because signals will get delivered to other threads and Python would forward
      # them for delayed handling to the main thread which hangs.
      # See CPython signalmodule.c.
      # Currently the best solution I can think of:
      while thread_obj.is_alive():
        join_orig(thread_obj, timeout=0.1)
    elif thread.get_ident() == main_thread_id and timeout > 0.1:
      # Limit the timeout. This should not matter for the underlying code.
      join_orig(thread_obj, timeout=0.1)
    else:
      # In all other cases, we can use the original.
      join_orig(thread_obj, timeout=timeout)
  threading.Thread.join = join_hacked

  # Mostly the same for Condition.wait().
  if PY3:
    # https://youtrack.jetbrains.com/issue/PY-34983
    # noinspection PyPep8Naming
    Condition = threading.Condition
  else:
    # noinspection PyUnresolvedReferences,PyPep8Naming,PyProtectedMember
    Condition = threading._Condition
  cond_wait_orig = Condition.wait

  # noinspection PyUnusedLocal
  def cond_wait_hacked(cond, timeout=None, *args):
    """
    :param Condition cond:
    :param float|None timeout:
    :param args:
    :rtype: bool
    """
    if thread.get_ident() == main_thread_id:
      if timeout is None:
        # Use a timeout anyway. This should not matter for the underlying code.
        return cond_wait_orig(cond, timeout=0.1)  # noqa  # https://youtrack.jetbrains.com/issue/PY-43915
      # There is some code (e.g. multiprocessing.pool) which relies on that
      # we respect the real specified timeout.
      # However, we cannot do multiple repeated calls to cond_wait_orig as we might miss the condition notify.
      # But in some Python versions, the underlying cond_wait_orig will anyway also use sleep.
      return cond_wait_orig(cond, timeout=timeout)  # noqa
    else:
      return cond_wait_orig(cond, timeout=timeout)  # noqa
  Condition.wait = cond_wait_hacked

  # And the same for Lock.acquire, very similar to Condition.wait.
  # However: can't set attributes of built-in/extension type 'thread.lock'.
  # We could wrap the whole threading.Lock, but that is too annoying for me now...
  # noinspection PyPep8Naming
  Lock = None
  if Lock:
    lock_acquire_orig = Lock.acquire  # noqa

    # Note: timeout argument was introduced in Python 3.
    def lock_acquire_hacked(lock, blocking=True, timeout=-1):
      """
      :param threading.Lock lock:
      :param bool blocking:
      :param float timeout:
      :rtype: bool
      """
      if not blocking:
        return lock_acquire_orig(lock, blocking=False)  # no timeout if not blocking
      # Everything is blocking now.
      if thread.get_ident() == main_thread_id:
        if timeout is None or timeout < 0:  # blocking without timeout
          if PY3:
            while not lock_acquire_orig(lock, blocking=True, timeout=0.1):
              pass
            return True
          else:  # Python 2. cannot use timeout
            while not lock_acquire_orig(lock, blocking=False):
              time.sleep(0.1)
            return True
        else:  # timeout is set. (Can only be with Python 3.)
          # Use a capped timeout. This should not matter for the underlying code.
          return lock_acquire_orig(lock, blocking=True, timeout=min(timeout, 0.1))
      # Fallback to default.
      if PY3:
        return lock_acquire_orig(lock, blocking=True, timeout=timeout)
      return lock_acquire_orig(lock, blocking=True)
    Lock.acquire = lock_acquire_hacked


def start_daemon_thread(target, args=()):
  """
  :param ()->None target:
  :param tuple args:
  :return: nothing
  """
  from threading import Thread
  t = Thread(target=target, args=args)
  t.daemon = True
  t.start()


def is_quitting():
  """
  :return: whether we are currently quitting (via :func:`rnn.finalize`)
  :rtype: bool
  """
  import returnn.__main__ as rnn
  if rnn.quit_returnn:  # via rnn.finalize()
    return True
  if getattr(sys, "exited", False):  # set via Debug module when an unexpected SIGINT occurs, or here
    return True
  return False


def interrupt_main():
  """
  Sends :class:`KeyboardInterrupt` to the main thread.

  :return: nothing
  """
  # noinspection PyProtectedMember,PyUnresolvedReferences
  is_main_thread = isinstance(threading.currentThread(), threading._MainThread)
  if is_quitting():  # ignore if we are already quitting
    if is_main_thread:  # strange to get again in main thread
      raise Exception("interrupt_main() from main thread while already quitting")
    # Not main thread. This will just exit the thread.
    sys.exit(1)
  sys.exited = True  # Don't do it twice.
  # noinspection PyProtectedMember,PyUnresolvedReferences
  sys.exited_frame = sys._getframe()
  if is_main_thread:
    raise KeyboardInterrupt
  else:
    thread.interrupt_main()
    sys.exit(1)  # And exit the thread.


class AsyncThreadRun(threading.Thread):
  """
  Daemon thread, wrapping some function ``func`` via :func:`wrap_async_func`.
  """

  def __init__(self, name, func):
    """
    :param str name:
    :param ()->T func:
    """
    super(AsyncThreadRun, self).__init__(name=name, target=self.main)
    self.func = func
    self.result = None
    self.daemon = True
    self.start()

  def main(self):
    """
    Thread target function.

    :return: nothing, will just set self.result
    """
    self.result = wrap_async_func(self.func)

  def get(self):
    """
    :return: joins the thread, and then returns the result
    :rtype: T
    """
    self.join()
    return self.result


def wrap_async_func(f):
  """
  Calls ``f()`` and returns the result.
  Wrapped up with catching all exceptions, printing stack trace, and :func:`interrupt_main`.

  :param ()->T f:
  :rtype: T
  """
  # noinspection PyBroadException
  try:
    from returnn.util import better_exchook
    better_exchook.install()
    return f()
  except Exception:
    sys.excepthook(*sys.exc_info())
    interrupt_main()


def try_run(func, args=(), catch_exc=Exception, default=None):
  """
  :param ((X)->T) func:
  :param tuple args:
  :param type[Exception] catch_exc:
  :param T2 default:
  :return: either ``func()`` or ``default`` if there was some exception
  :rtype: T|T2
  """
  # noinspection PyBroadException
  try:
    return func(*args)
  except catch_exc:
    return default


def class_idx_seq_to_1_of_k(seq, num_classes):
  """
  Basically one_hot.

  :param list[int]|np.ndarray seq:
  :param int num_classes:
  :rtype: np.ndarray
  """
  num_frames = len(seq)
  m = np.zeros((num_frames, num_classes), dtype='float32')
  m[np.arange(num_frames), seq] = 1
  return m


def uniq(seq):
  """
  Like Unix tool uniq. Removes repeated entries.

  :param seq: numpy.array
  :return: seq
  """
  diffs = np.ones_like(seq)
  diffs[1:] = seq[1:] - seq[:-1]
  idx = diffs.nonzero()
  return seq[idx]


def slice_pad_zeros(x, begin, end, axis=0):
  """
  :param numpy.ndarray x: of shape (..., time, ...)
  :param int begin:
  :param int end:
  :param int axis:
  :return: basically x[begin:end] (with axis==0) but if begin < 0 or end > x.shape[0],
   it will not discard these frames but pad zeros, such that the resulting shape[0] == end - begin.
  :rtype: numpy.ndarray
  """
  assert axis == 0, "not yet fully implemented otherwise"
  pad_left, pad_right = 0, 0
  if begin < 0:
    pad_left = -begin
    begin = 0
  elif begin >= x.shape[axis]:
    return np.zeros((end - begin,) + x.shape[1:], dtype=x.dtype)
  assert end >= begin
  if end > x.shape[axis]:
    pad_right = end - x.shape[axis]
    end = x.shape[axis]
  return np.pad(x[begin:end], [(pad_left, pad_right)] + [(0, 0)] * (x.ndim - 1), mode="constant")


def random_orthogonal(shape, gain=1., seed=None):
  """
  Returns a random orthogonal matrix of the given shape.
  Code borrowed and adapted from Keras: https://github.com/fchollet/keras/blob/master/keras/initializers.py
  Reference: Saxe et al., http://arxiv.org/abs/1312.6120
  Related: Unitary Evolution Recurrent Neural Networks, https://arxiv.org/abs/1511.06464

  :param tuple[int] shape:
  :param float gain:
  :param int seed: for Numpy random generator
  :return: random orthogonal matrix
  :rtype: numpy.ndarray
  """
  num_rows = 1
  for dim in shape[:-1]:
    num_rows *= dim
  num_cols = shape[-1]
  flat_shape = (num_rows, num_cols)
  if seed is not None:
    rnd = np.random.RandomState(seed=seed)
  else:
    rnd = np.random
  a = rnd.normal(0.0, 1.0, flat_shape)
  u, _, v = np.linalg.svd(a, full_matrices=False)
  # Pick the one with the correct shape.
  q = u if u.shape == flat_shape else v
  q = q.reshape(shape)
  return gain * q[:shape[0], :shape[1]]


_have_inplace_increment = None
_native_inplace_increment = None  # type: typing.Optional[typing.Callable[[np.ndarray,np.ndarray,np.ndarray],np.ndarray]]  # nopep8


def inplace_increment(x, idx, y):
  """
  This basically does `x[idx] += y`.
  The difference to the Numpy version is that in case some index is there multiple
  times, it will only be incremented once (and it is not specified which one).
  See also theano.tensor.subtensor.AdvancedIncSubtensor documentation.

  :param numpy.ndarray x:
  :param numpy.ndarray idx:
  :param numpy.ndarray y:
  :rtype: numpy.ndarray
  """
  # https://youtrack.jetbrains.com/issue/PY-28498
  global _have_inplace_increment, inplace_increment, _native_inplace_increment  # noqa
  if _have_inplace_increment is None:
    native_inpl_incr = None
    # noinspection PyPackageRequirements,PyUnresolvedReferences
    import theano
    if theano.config.cxx:
      # noinspection PyPackageRequirements,PyUnresolvedReferences
      import theano.gof.cutils  # needed to import cutils_ext
      try:
        from cutils_ext.cutils_ext import inplace_increment as native_inpl_incr  # noqa
      except ImportError:
        pass
    if native_inpl_incr:
      _have_inplace_increment = True
      _native_inplace_increment = native_inpl_incr
      inplace_increment = native_inpl_incr  # replace myself
      return inplace_increment(x, idx, y)  # noqa
    _have_inplace_increment = False
  if _have_inplace_increment is True:
    return _native_inplace_increment(x, idx, y)  # noqa
  raise NotImplementedError("need Numpy 1.8 or later")


def prod(ls):
  """
  :param list[T]|tuple[T]|numpy.ndarray ls:
  :rtype: T|int|float
  """
  if len(ls) == 0:
    return 1
  x = ls[0]
  for y in ls[1:]:
    x *= y
  return x


def parse_orthography_into_symbols(orthography, upper_case_special=True, word_based=False,
                                   square_brackets_for_specials=True):
  """
  For Speech.
  Example:
    orthography = "hello [HESITATION] there "
    with word_based == False: returns list("hello ") + ["[HESITATION]"] + list(" there ").
    with word_based == True: returns ["hello", "[HESITATION]", "there"]
  No pre/post-processing such as:
  Spaces are kept as-is. No stripping at begin/end. (E.g. trailing spaces are not removed.)
  No tolower/toupper.
  Doesn't add [BEGIN]/[END] symbols or so.
  Any such operations should be done explicitly in an additional function.
  Anything in []-brackets are meant as special-symbols.
  Also see parse_orthography() which includes some preprocessing.

  :param str orthography: example: "hello [HESITATION] there "
  :param bool upper_case_special: whether the special symbols are always made upper case
  :param bool word_based: whether we split on space and return full words
  :param bool square_brackets_for_specials: handle "[...]"
  :rtype: list[str]
  """
  ret = []
  in_special = 0
  for c in orthography:
    if in_special:
      if c == "[":  # special-special
        in_special += 1
        ret[-1] += "["
      elif c == "]":
        in_special -= 1
        ret[-1] += "]"
      elif upper_case_special:
        ret[-1] += c.upper()
      else:
        ret[-1] += c
    else:  # not in_special
      if square_brackets_for_specials and c == "[":
        in_special = 1
        ret += ["["]
      else:
        if word_based:
          if c.isspace():
            ret += [""]
          else:
            if not ret:
              ret += [""]
            ret[-1] += c
        else:  # not word_based
          ret += c
  return ret


def parse_orthography(orthography, prefix=(), postfix=("[END]",),
                      remove_chars="(){}", collapse_spaces=True, final_strip=True,
                      **kwargs):
  """
  For Speech. Full processing.
  Example:
    orthography = "hello [HESITATION] there "
    with word_based == False: returns list("hello ") + ["[HESITATION]"] + list(" there") + ["[END]"]
    with word_based == True: returns ["hello", "[HESITATION]", "there", "[END]"]
  Does some preprocessing on orthography and then passes it on to parse_orthography_into_symbols().

  :param str orthography: e.g. "hello [HESITATION] there "
  :param list[str] prefix: will add this prefix
  :param list[str] postfix: will add this postfix
  :param str remove_chars: those chars will just be removed at the beginning
  :param bool collapse_spaces: whether multiple spaces and tabs are collapsed into a single space
  :param bool final_strip: whether we strip left and right
  :param kwargs: passed on to parse_orthography_into_symbols()
  :rtype: list[str]
  """
  for c in remove_chars:
    orthography = orthography.replace(c, "")
  if collapse_spaces:
    orthography = " ".join(orthography.split())
  if final_strip:
    orthography = orthography.strip()
  return list(prefix) + parse_orthography_into_symbols(orthography, **kwargs) + list(postfix)


def json_remove_comments(string, strip_space=True):
  """
  :type string: str
  :param bool strip_space:
  :rtype: str

  via https://github.com/getify/JSON.minify/blob/master/minify_json.py,
  by Gerald Storer, Pradyun S. Gedam, modified by us.
  """
  tokenizer = re.compile('"|(/\\*)|(\\*/)|(//)|\n|\r')
  end_slashes_re = re.compile(r'(\\)*$')

  in_string = False
  in_multi = False
  in_single = False

  new_str = []
  index = 0

  for match in re.finditer(tokenizer, string):

    if not (in_multi or in_single):
      tmp = string[index:match.start()]
      if not in_string and strip_space:
        # replace white space as defined in standard
        tmp = re.sub('[ \t\n\r]+', '', tmp)
      new_str.append(tmp)

    index = match.end()
    val = match.group()

    if val == '"' and not (in_multi or in_single):
      escaped = end_slashes_re.search(string, 0, match.start())

      # start of string or unescaped quote character to end string
      if not in_string or (escaped is None or len(escaped.group()) % 2 == 0):
        in_string = not in_string
      index -= 1  # include " character in next catch
    elif not (in_string or in_multi or in_single):
      if val == '/*':
        in_multi = True
      elif val == '//':
        in_single = True
    elif val == '*/' and in_multi and not (in_string or in_single):
      in_multi = False
    elif val in '\r\n' and not (in_multi or in_string) and in_single:
      in_single = False
    elif not ((in_multi or in_single) or (val in ' \r\n\t' and strip_space)):
      new_str.append(val)

  new_str.append(string[index:])
  return ''.join(new_str)


def _py2_unicode_to_str_recursive(s):
  """
  This is supposed to be run with Python 2.
  Also see :func:`as_str` and :func:`py2_utf8_str_to_unicode`.

  :param str|unicode s: or any recursive structure such as dict, list, tuple
  :return: Python 2 str (is like Python 3 UTF-8 formatted bytes)
  :rtype: str
  """
  if isinstance(s, dict):
    return {_py2_unicode_to_str_recursive(key): _py2_unicode_to_str_recursive(value) for key, value in s.items()}
  elif isinstance(s, (list, tuple)):
    return make_seq_of_type(type(s), [_py2_unicode_to_str_recursive(element) for element in s])
  elif isinstance(s, unicode):
    return s.encode('utf-8')  # Python 2 str, Python 3 bytes
  else:
    return s


def load_json(filename=None, content=None):
  """
  :param str|None filename:
  :param str|None content:
  :rtype: dict[str]
  """
  if content:
    assert not filename
  else:
    content = open(filename).read()
  import json
  content = json_remove_comments(content)
  try:
    json_content = json.loads(content)
  except ValueError as e:
    raise Exception("config looks like JSON but invalid json content, %r" % e)
  if not PY3:
    json_content = _py2_unicode_to_str_recursive(json_content)
  return json_content


class NumbersDict:
  """
  It's mostly like dict[str,float|int] & some optional broadcast default value.
  It implements the standard math bin ops in a straight-forward way.
  """

  def __init__(self, auto_convert=None, numbers_dict=None, broadcast_value=None):
    """
    :param dict|NumbersDict|T auto_convert: first argument, so that we can automatically convert/copy
    :param dict numbers_dict:
    :param T broadcast_value:
    """
    if auto_convert is not None:
      assert broadcast_value is None
      assert numbers_dict is None
      if isinstance(auto_convert, dict):
        numbers_dict = auto_convert
      elif isinstance(auto_convert, NumbersDict):
        numbers_dict = auto_convert.dict
        broadcast_value = auto_convert.value
      else:
        broadcast_value = auto_convert
    if numbers_dict is None:
      numbers_dict = {}
    else:
      numbers_dict = dict(numbers_dict)  # force copy

    self.dict = numbers_dict
    self.value = broadcast_value
    self.max = self.__max_error

  def copy(self):
    """
    :rtype: NumbersDict
    """
    return NumbersDict(self)

  @classmethod
  def constant_like(cls, const_number, numbers_dict):
    """
    :param int|float|object const_number:
    :param NumbersDict numbers_dict:
    :return: NumbersDict with same keys as numbers_dict
    :rtype: NumbersDict
    """
    return NumbersDict(
      broadcast_value=const_number if (numbers_dict.value is not None) else None,
      numbers_dict={k: const_number for k in numbers_dict.dict.keys()})

  def copy_like(self, numbers_dict):
    """
    :param NumbersDict numbers_dict:
    :return: copy of self with same keys as numbers_dict as far as we have them
    :rtype: NumbersDict
    """
    if self.value is not None:
      return NumbersDict(
        broadcast_value=self.value if (numbers_dict.value is not None) else None,
        numbers_dict={k: self[k] for k in numbers_dict.dict.keys()})
    else:
      return NumbersDict(
        broadcast_value=None,
        numbers_dict={k: self[k] for k in numbers_dict.dict.keys() if k in self.dict})

  @property
  def keys_set(self):
    """
    :rtype: set[str]
    """
    return set(self.dict.keys())

  def __getitem__(self, key):
    if self.value is not None:
      return self.dict.get(key, self.value)
    return self.dict[key]

  def __setitem__(self, key, value):
    self.dict[key] = value

  def __delitem__(self, key):
    del self.dict[key]

  def get(self, key, default=None):
    """
    :param str key:
    :param T default:
    :rtype: object|T
    """
    # Keep consistent with self.__get_item__. If self.value is set, this will always be the default value.
    return self.dict.get(key, self.value if self.value is not None else default)

  def pop(self, key, *args):
    """
    :param str key:
    :param T args: default, or not
    :rtype: object|T
    """
    return self.dict.pop(key, *args)

  def __iter__(self):
    # This can potentially cause confusion. So enforce explicitness.
    # For a dict, we would return the dict keys here.
    # Also, max(self) would result in a call to self.__iter__(),
    # which would only make sense for our values, not the dict keys.
    raise Exception("%s.__iter__ is undefined" % self.__class__.__name__)

  def keys(self):
    """
    :rtype: set[str]
    """
    return self.dict.keys()

  def values(self):
    """
    :rtype: list[object]
    """
    return list(self.dict.values()) + ([self.value] if self.value is not None else [])

  def items(self):
    """
    :return: dict items. this excludes self.value
    :rtype: str[(str,object)]
    """
    return self.dict.items()

  def has_values(self):
    """
    :rtype: bool
    """
    return bool(self.dict) or self.value is not None

  def unary_op(self, op):
    """
    :param (T)->T2 op:
    :return: new NumbersDict, where ``op`` is applied on all values
    :rtype: NumbersDict
    """
    res = NumbersDict()
    if self.value is not None:
      res.value = op(self.value)
    for k, v in self.dict.items():
      res.dict[k] = op(v)
    return res

  @classmethod
  def bin_op_scalar_optional(cls, self, other, zero, op):
    """
    :param T self:
    :param T other:
    :param T zero:
    :param (T,T)->T op:
    :rtype: T
    """
    if self is None and other is None:
      return None
    if self is None:
      self = zero
    if other is None:
      other = zero
    return op(self, other)

  @classmethod
  def bin_op(cls, self, other, op, zero, result=None):
    """
    :param NumbersDict|int|float|T self:
    :param NumbersDict|int|float|T other:
    :param (T,T)->T op:
    :param T zero:
    :param NumbersDict|None result:
    :rtype: NumbersDict
    """
    if not isinstance(self, NumbersDict):
      if isinstance(other, NumbersDict):
        self = NumbersDict.constant_like(self, numbers_dict=other)
      else:
        self = NumbersDict(self)
    if not isinstance(other, NumbersDict):
      other = NumbersDict.constant_like(other, numbers_dict=self)
    if result is None:
      result = NumbersDict()
    assert isinstance(result, NumbersDict)
    for k in self.keys_set | other.keys_set:
      result[k] = cls.bin_op_scalar_optional(self.get(k, None), other.get(k, None), zero=zero, op=op)
    result.value = cls.bin_op_scalar_optional(self.value, other.value, zero=zero, op=op)
    return result

  def __add__(self, other):
    return self.bin_op(self, other, op=lambda a, b: a + b, zero=0)

  __radd__ = __add__

  def __iadd__(self, other):
    return self.bin_op(self, other, op=lambda a, b: a + b, zero=0, result=self)

  def __sub__(self, other):
    return self.bin_op(self, other, op=lambda a, b: a - b, zero=0)

  def __rsub__(self, other):
    return self.bin_op(self, other, op=lambda a, b: b - a, zero=0)

  def __isub__(self, other):
    return self.bin_op(self, other, op=lambda a, b: a - b, zero=0, result=self)

  def __mul__(self, other):
    return self.bin_op(self, other, op=lambda a, b: a * b, zero=1)

  __rmul__ = __mul__

  def __imul__(self, other):
    return self.bin_op(self, other, op=lambda a, b: a * b, zero=1, result=self)

  def __div__(self, other):
    return self.bin_op(self, other, op=lambda a, b: a / b, zero=1)

  __rdiv__ = __div__
  __truediv__ = __div__

  def __idiv__(self, other):
    return self.bin_op(self, other, op=lambda a, b: a / b, zero=1, result=self)

  __itruediv__ = __idiv__

  def __floordiv__(self, other):
    return self.bin_op(self, other, op=lambda a, b: a // b, zero=1)

  def __ifloordiv__(self, other):
    return self.bin_op(self, other, op=lambda a, b: a // b, zero=1, result=self)

  def __neg__(self):
    return self.unary_op(op=lambda a: -a)

  def __bool__(self):
    return any(self.values())

  __nonzero__ = __bool__  # Python 2

  def elem_eq(self, other, result_with_default=True):
    """
    Element-wise equality check with other.
    Note about broadcast default value: Consider some key which is neither in self nor in other.
      This means that self[key] == self.default, other[key] == other.default.
      Thus, in case that self.default != other.default, we get res.default == False.
      Then, all(res.values()) == False, even when all other values are True.
      This is sometimes not what we want.
      You can control the behavior via result_with_default.

    :param NumbersDict|T other:
    :param bool result_with_default:
    :rtype: NumbersDict
    """
    def op(a, b):
      """
      :param a:
      :param b:
      :rtype: bool|None
      """
      if a is None:
        return None
      if b is None:
        return None
      return a == b

    res = self.bin_op(self, other, op=op, zero=None)
    if not result_with_default:
      res.value = None
    return res

  def __eq__(self, other):
    """
    :param NumbersDict|T other:
    :return: whether self == other elemwise. see self.elem_eq
    :rtype: bool
    """
    return all(self.elem_eq(other).values())

  def __ne__(self, other):
    """
    :param NumbersDict|T other:
    :return: not (self == other)
    :rtype: bool
    """
    return not (self == other)

  def __cmp__(self, other):
    # There is no good straight-forward implementation
    # and it would just confuse.
    raise Exception("%s.__cmp__ is undefined" % self.__class__.__name__)

  def any_compare(self, other, cmp):
    """
    :param NumbersDict other:
    :param ((object,object)->True) cmp:
    :rtype: True
    """
    for key in self.keys():
      if key in other.keys():
        if cmp(self[key], other[key]):
          return True
      elif other.value is not None:
        if cmp(self[key], other.value):
          return True
    if self.value is not None and other.value is not None:
      if cmp(self.value, other.value):
        return True
    return False

  @staticmethod
  def _max(*args):
    args = [a for a in args if a is not None]
    if not args:
      return None
    if len(args) == 1:
      return args[0]
    return max(*args)

  @staticmethod
  def _min(*args):
    args = [a for a in args if a is not None]
    if not args:
      return None
    if len(args) == 1:
      return args[0]
    return min(*args)

  @classmethod
  def max(cls, items):
    """
    Element-wise maximum for item in items.
    :param list[NumbersDict|int|float] items:
    :rtype: NumbersDict
    """
    assert items
    if len(items) == 1:
      return NumbersDict(items[0])
    if len(items) == 2:
      return cls.bin_op(items[0], items[1], op=cls._max, zero=None)
    return cls.max([items[0], cls.max(items[1:])])

  @classmethod
  def min(cls, items):
    """
    Element-wise minimum for item in items.
    :param list[NumbersDict|int|float] items:
    :rtype: NumbersDict
    """
    assert items
    if len(items) == 1:
      return NumbersDict(items[0])
    if len(items) == 2:
      return cls.bin_op(items[0], items[1], op=cls._min, zero=None)
    return cls.min([items[0], cls.min(items[1:])])

  @staticmethod
  def __max_error():
    # Will replace self.max for each instance. To be sure that we don't confuse it with self.max_value.
    raise Exception("Use max_value instead.")

  def max_value(self):
    """
    Maximum of our values.
    """
    return max(self.values())

  def min_value(self):
    """
    Minimum of our values.
    """
    return min(self.values())

  def __repr__(self):
    if self.value is None and not self.dict:
      return "%s()" % self.__class__.__name__
    if self.value is None and self.dict:
      return "%s(%r)" % (self.__class__.__name__, self.dict)
    if not self.dict and self.value is not None:
      return "%s(%r)" % (self.__class__.__name__, self.value)
    return "%s(numbers_dict=%r, broadcast_value=%r)" % (
           self.__class__.__name__, self.dict, self.value)


def collect_class_init_kwargs(cls, only_with_default=False):
  """
  :param type cls: class, where it assumes that kwargs are passed on to base classes
  :param bool only_with_default: if given will only return the kwargs with default values
  :return: set if not with_default, otherwise the dict to the default values
  :rtype: list[str] | dict[str]
  """
  from collections import OrderedDict
  if only_with_default:
    kwargs = OrderedDict()
  else:
    kwargs = []
  for cls_ in inspect.getmro(cls):
    # Check Python function. Could be builtin func or so. Python 2 getargspec does not work in that case.
    if not inspect.ismethod(cls_.__init__) and not inspect.isfunction(cls_.__init__):
      continue
    arg_spec = getargspec(cls_.__init__)
    args = arg_spec.args[1:]  # first arg is self, ignore
    if only_with_default:
      if arg_spec.defaults:
        assert len(arg_spec.defaults) <= len(args)
        args = args[len(args) - len(arg_spec.defaults):]
        assert len(arg_spec.defaults) == len(args), arg_spec
        for arg, default in zip(args, arg_spec.defaults):
          kwargs[arg] = default
    else:
      for arg in args:
        if arg not in kwargs:
          kwargs.append(arg)
  return kwargs


def getargspec(func):
  """
  :func:`inspect.getfullargspec` or `inspect.getargspec` (Python 2)

  :param func:
  :return: FullArgSpec
  """
  if PY3:
    return inspect.getfullargspec(func)
  else:
    # noinspection PyDeprecation
    return inspect.getargspec(func)


def collect_mandatory_class_init_kwargs(cls):
  """
  :param type cls:
  :return: list of kwargs which have no default, i.e. which must be provided
  :rtype: list[str]
  """
  all_kwargs = collect_class_init_kwargs(cls, only_with_default=False)
  default_kwargs = collect_class_init_kwargs(cls, only_with_default=True)
  mandatory_kwargs = []
  for arg in all_kwargs:
    if arg not in default_kwargs:
      mandatory_kwargs.append(arg)
  return mandatory_kwargs


def help_on_type_error_wrong_args(cls, kwargs):
  """
  :param type cls:
  :param list[str] kwargs:
  """
  mandatory_args = collect_mandatory_class_init_kwargs(cls)
  for arg in kwargs:
    if arg in mandatory_args:
      mandatory_args.remove(arg)
  all_kwargs = collect_class_init_kwargs(cls)
  unknown_args = []
  for arg in kwargs:
    if arg not in all_kwargs:
      unknown_args.append(arg)
  if mandatory_args or unknown_args:
    print("Args mismatch? Missing are %r, unknowns are %r. Kwargs %r." % (mandatory_args, unknown_args, kwargs))


def custom_exec(source, source_filename, user_ns, user_global_ns):
  """
  :param str source:
  :param str source_filename:
  :param dict[str] user_ns:
  :param dict[str] user_global_ns:
  :return: nothing
  """
  if not source.endswith("\n"):
    source += "\n"
  co = compile(source, source_filename, "exec")
  user_global_ns["__package__"] = "returnn"  # important so that imports work
  eval(co, user_global_ns, user_ns)


class FrozenDict(dict):
  """
  Frozen dict.
  """

  def __setitem__(self, key, value):
    raise ValueError("FrozenDict cannot be modified")

  def __hash__(self):
    return hash(tuple(sorted(self.items())))


def make_hashable(obj):
  """
  Theano needs hashable objects in some cases, e.g. the properties of Ops.
  This converts all objects as such, i.e. into immutable frozen types.

  :param T|dict|list|tuple obj:
  :rtype: T|FrozenDict|tuple
  """
  if isinstance(obj, dict):
    return FrozenDict([make_hashable(item) for item in obj.items()])
  elif isinstance(obj, (list, tuple)):
    return tuple([make_hashable(item) for item in obj])
  elif isinstance(obj, (str, unicode, float, int, long)):
    return obj
  elif obj is None:
    return obj
  else:
    assert False, "don't know how to make hashable: %r (%r)" % (obj, type(obj))


def make_dll_name(basename):
  """
  :param str basename:
  :return: e.g. "lib%s.so" % basename, depending on sys.platform
  :rtype: str
  """
  if sys.platform == "darwin":
    return "lib%s.dylib" % basename
  elif sys.platform == "win32":
    return "%s.dll" % basename
  else:  # Linux, Unix
    return "lib%s.so" % basename


def escape_c_str(s):
  """
  :param str s:
  :return: C-escaped str
  :rtype: str
  """
  return '"%s"' % s.replace("\\\\", "\\").replace("\n", "\\n").replace("\"", "\\\"").replace("'", "\\'")


def attr_chain(base, attribs):
  """
  :param object base:
  :param list[str]|tuple[str]|str attribs:
  :return: getattr(getattr(object, attribs[0]), attribs[1]) ...
  :rtype: object
  """
  if not isinstance(attribs, (list, tuple)):
    assert isinstance(attribs, str)
    attribs = [attribs]
  else:
    attribs = list(attribs)
  for i in range(len(attribs)):
    base = getattr(base, attribs[i])
  return base


def to_bool(v):
  """
  :param int|float|str v: if it is a string, it should represent some integer, or alternatively "true" or "false"
  :rtype: bool
  """
  try:
    return bool(int(v))
  except ValueError:
    pass
  if isinstance(v, (str, unicode)):
    v = v.lower()
    if v in ["true", "yes", "on", "1"]:
      return True
    if v in ["false", "no", "off", "0"]:
      return False
  raise ValueError("to_bool cannot handle %r" % v)


def as_str(s):
  """
  :param str|unicode|bytes s:
  :rtype: str|unicode
  """
  if isinstance(s, str) or 'unicode' in str(type(s)):
    return s
  if isinstance(s, bytes) or isinstance(s, unicode):
    # noinspection PyUnresolvedReferences
    return s.decode("utf8")
  assert False, "unknown type %s" % type(s)


def py2_utf8_str_to_unicode(s):
  """
  :param str s: e.g. the string literal "äöü" in Python 3 is correct, but in Python 2 it should have been u"äöü",
    but just using "äöü" will actually be the raw utf8 byte sequence.
    This can happen when you eval() some string.
    We assume that you are using Python 2, and got the string (not unicode object) "äöü", or maybe "abc".
    Also see :func:`_py2_unicode_to_str_recursive` and :func:`as_str`.
  :return: if it is indeed unicode, it will return the unicode object, otherwise it keeps the string
  :rtype: str|unicode
  """
  assert not PY3
  assert isinstance(s, str)
  try:
    # noinspection PyUnresolvedReferences
    s.decode("ascii")
    return s
  except UnicodeDecodeError:
    pass
  # noinspection PyUnresolvedReferences
  return s.decode("utf8")


def unicode_to_str(s):
  """
  The behavior is different depending on Python 2 or Python 3. In all cases, the returned type is a str object.
  Python 2:
    We return the utf8 encoded str (which is like Python 3 bytes, or for ASCII, there is no difference).
  Python 3:
    We return a str object.
  Note that this function probably does not make much sense.
  It might be used when there is other code which expects a str object, no matter if Python 2 or Python 3.
  In Python 2, a str object often holds UTF8 text, so the behavior of this function is fine then.
  Also see :func:`as_str`.

  :param str|unicode|bytes s:
  :rtype: str
  """
  if PY3 and isinstance(s, bytes):
    s = s.decode("utf8")
    assert isinstance(s, str)
  if not PY3 and isinstance(s, unicode):
    s = s.encode("utf8")
    assert isinstance(s, str)
  assert isinstance(s, str)
  return s


def deepcopy(x, stop_types=None):
  """
  Simpler variant of copy.deepcopy().
  Should handle some edge cases as well, like copying module references.

  :param T x: an arbitrary object
  :param list[type]|None stop_types: objects of these types will not be deep-copied, only the reference is passed
  :rtype: T
  """
  # See also class Pickler from TaskSystem.
  # Or: https://mail.python.org/pipermail/python-ideas/2013-July/021959.html
  from .task_system import Pickler, Unpickler

  persistent_memo = {}  # id -> obj

  def persistent_id(obj):
    """
    :param object obj:
    :rtype: int|None
    """
    if stop_types and isinstance(obj, tuple(stop_types)):
      persistent_memo[id(obj)] = obj
      return id(obj)
    return None

  def pickle_dumps(obj):
    """
    :param object obj:
    :rtype: bytes
    """
    sio = BytesIO()
    p = Pickler(sio)
    p.persistent_id = persistent_id
    p.dump(obj)
    return sio.getvalue()

  # noinspection PyShadowingNames
  def pickle_loads(s):
    """
    :param bytes s:
    :rtype: object
    """
    p = Unpickler(BytesIO(s))
    p.persistent_load = persistent_memo.__getitem__
    return p.load()

  s = pickle_dumps(x)
  c = pickle_loads(s)
  return c


def load_txt_vector(filename):
  """
  Expect line-based text encoding in file.
  We also support Sprint XML format, which has some additional xml header and footer,
  which we will just strip away.

  :param str filename:
  :rtype: list[float]
  """
  return [float(line) for line in open(filename).read().splitlines() if line and not line.startswith("<")]


class CollectionReadCheckCovered:
  """
  Wraps around a dict. It keeps track about all the keys which were read from the dict.
  Via :func:`assert_all_read`, you can check that there are no keys in the dict which were not read.
  The usage is for config dict options, where the user has specified a range of options,
  and where in the code there is usually a default for every non-specified option,
  to check whether all the user-specified options are also used (maybe the user made a typo).
  """

  def __init__(self, collection, truth_value=None):
    """
    :param dict[str] collection:
    :param None|bool truth_value: note: check explicitly for self.truth_value, bool(self) is not the same!
    """
    self.collection = collection
    if truth_value is None:
      truth_value = bool(self.collection)
    self.truth_value = truth_value
    self.got_items = set()

  def __repr__(self):
    return "%s(%r, truth_value=%r)" % (self.__class__.__name__, self.collection, self.truth_value)

  @classmethod
  def from_bool_or_dict(cls, value):
    """
    :param bool|dict[str] value:
    :rtype: CollectionReadCheckCovered
    """
    if isinstance(value, bool):
      return cls(collection={}, truth_value=value)
    if isinstance(value, dict):
      return cls(collection=value)
    raise TypeError("invalid type: %s" % type(value))

  def __getitem__(self, item):
    res = self.collection[item]
    self.got_items.add(item)
    return res

  def get(self, item, default=None):
    """
    :param str item:
    :param T default:
    :rtype: T|object|None
    """
    try:
      return self[item]
    except KeyError:
      return default

  def __bool__(self):  # Python 3
    return self.truth_value

  __nonzero__ = __bool__  # Python 2

  def __len__(self):
    return len(self.collection)

  def __iter__(self):
    for k in self.collection:
      yield self[k]

  def assert_all_read(self):
    """
    Asserts that all items have been read.
    """
    remaining = set(self.collection).difference(self.got_items)
    assert not remaining, "The keys %r were not read in the collection %r." % (remaining, self.collection)


def which(program):
  """
  Finds `program` in some of the dirs of the PATH env var.

  :param str program: e.g. "python"
  :return: full path, e.g. "/usr/bin/python", or None
  :rtype: str|None
  """
  # noinspection PyShadowingNames
  def is_exe(path):
    """
    :param str path:
    :rtype: str
    """
    return os.path.isfile(path) and os.access(path, os.X_OK)

  fpath, fname = os.path.split(program)
  if fpath:
    if is_exe(program):
      return program
  else:
    for path in os.environ["PATH"].split(os.pathsep):
      path = path.strip('"')
      exe_file = os.path.join(path, program)
      if is_exe(exe_file):
        return exe_file

  return None


def which_pip():
  """
  :rtype: str
  :return: path to pip for the current Python env
  """
  # Before we look anywhere in PATH, check if there is some pip alongside to the Python executable.
  # This might be more reliable.
  py = sys.executable
  dir_name, basename = py.rsplit("/", 1)
  if basename.startswith("python"):
    postfix = basename[len("python"):]
    pip_path = "%s/pip%s" % (dir_name, postfix)
    if os.path.exists(pip_path):
      return pip_path
  # Generic fallback.
  pip_path = which("pip")
  return pip_path


def pip_install(*pkg_names):
  """
  Install packages via pip for the current Python env.

  :param str pkg_names:
  """
  py = sys.executable
  pip_path = which_pip()
  print("Pip install", *pkg_names)
  in_virtual_env = hasattr(sys, 'real_prefix')  # https://stackoverflow.com/questions/1871549/
  cmd = [py, pip_path, "install"]
  if not in_virtual_env:
    cmd += ["--user"]
  cmd += list(pkg_names)
  print("$ %s" % " ".join(cmd))
  subprocess.check_call(cmd, cwd="/")
  _pip_installed_packages.clear()  # force reload


_pip_installed_packages = set()


def pip_check_is_installed(pkg_name):
  """
  :param str pkg_name: without version, e.g. just "tensorflow" or "tensorflow-gpu"
  :rtype: bool
  """
  if not _pip_installed_packages:
    py = sys.executable
    pip_path = which_pip()
    cmd = [py, pip_path, "freeze"]
    for line in sys_exec_out(*cmd).splitlines():
      if line:
        _pip_installed_packages.add(line[:line.index("==")])
  return pkg_name in _pip_installed_packages


_original_execv = None
_original_execve = None
_original_execvpe = None


def overwrite_os_exec(prefix_args):
  """
  :param list[str] prefix_args:
  """
  global _original_execv, _original_execve, _original_execvpe
  if not _original_execv:
    _original_execv = os.execv
  if not _original_execve:
    _original_execve = os.execve
  if not _original_execvpe:
    # noinspection PyProtectedMember,PyUnresolvedReferences
    _original_execvpe = os._execvpe

  # noinspection PyUnusedLocal
  def wrapped_execvpe(file, args, env=None):
    """
    :param file:
    :param list[str]|tuple[str] args:
    :param dict[str] env:
    """
    new_args = prefix_args + [which(args[0])] + args[1:]
    sys.stderr.write("$ %s\n" % " ".join(new_args))
    sys.stderr.flush()
    _original_execvpe(file=prefix_args[0], args=new_args, env=env)

  def execv(path, args):
    """
    :param str path:
    :param list[str]|tuple[str] args:
    """
    if args[:len(prefix_args)] == prefix_args:
      _original_execv(path, args)
    else:
      wrapped_execvpe(path, args)

  def execve(path, args, env):
    """
    :param str path:
    :param list[str]|tuple[str] args:
    :param dict[str] env:
    """
    if args[:len(prefix_args)] == prefix_args:
      _original_execve(path, args, env)
    else:
      wrapped_execvpe(path, args, env)

  def execl(file, *args):
    """execl(file, *args)

    Execute the executable file with argument list args, replacing the
    current process. """
    os.execv(file, args)

  def execle(file, *args):
    """execle(file, *args, env)

    Execute the executable file with argument list args and
    environment env, replacing the current process. """
    env = args[-1]
    os.execve(file, args[:-1], env)

  def execlp(file, *args):
    """execlp(file, *args)

    Execute the executable file (which is searched for along $PATH)
    with argument list args, replacing the current process. """
    os.execvp(file, args)

  def execlpe(file, *args):
    """execlpe(file, *args, env)

    Execute the executable file (which is searched for along $PATH)
    with argument list args and environment env, replacing the current
    process. """
    env = args[-1]
    os.execvpe(file, args[:-1], env)

  def execvp(file, args):
    """execvp(file, args)

    Execute the executable file (which is searched for along $PATH)
    with argument list args, replacing the current process.
    args may be a list or tuple of strings. """
    wrapped_execvpe(file, args)

  def execvpe(file, args, env):
    """execvpe(file, args, env)

    Execute the executable file (which is searched for along $PATH)
    with argument list args and environment env , replacing the
    current process.
    args may be a list or tuple of strings. """
    wrapped_execvpe(file, args, env)

  os.execv = execv
  os.execve = execve
  os.execl = execl
  os.execle = execle
  os.execlp = execlp
  os.execlpe = execlpe
  os.execvp = execvp
  os.execvpe = execvpe
  os._execvpe = wrapped_execvpe


def get_lsb_release():
  """
  :return: ``/etc/lsb-release`` parsed as a dict
  :rtype: dict[str,str]
  """
  d = {}
  for line in open("/etc/lsb-release").read().splitlines():
    k, v = line.split("=", 1)
    if v[0] == v[-1] == "\"":
      v = v[1:-1]
    d[k] = v
  return d


def get_ubuntu_major_version():
  """
  :rtype: int|None
  """
  d = get_lsb_release()
  if d["DISTRIB_ID"] != "Ubuntu":
    return None
  return int(float(d["DISTRIB_RELEASE"]))


def auto_prefix_os_exec_prefix_ubuntu(prefix_args, ubuntu_min_version=16):
  """
  :param list[str] prefix_args:
  :param int ubuntu_min_version:

  Example usage:
    auto_prefix_os_exec_prefix_ubuntu(["/u/zeyer/tools/glibc217/ld-linux-x86-64.so.2"])
  """
  ubuntu_version = get_ubuntu_major_version()
  if ubuntu_version is None:
    return
  if ubuntu_version >= ubuntu_min_version:
    return
  print("You are running Ubuntu %i, thus we prefix all os.exec with %s." % (ubuntu_version, prefix_args))
  assert os.path.exists(prefix_args[0])
  overwrite_os_exec(prefix_args=prefix_args)


def cleanup_env_var_path(env_var, path_prefix):
  """
  :param str env_var: e.g. "LD_LIBRARY_PATH"
  :param str path_prefix:

  Will remove all paths in os.environ[env_var] which are prefixed with path_prefix.
  """
  if env_var not in os.environ:
    return
  ps = os.environ[env_var].split(":")

  def f(p):
    """
    :param str p:
    :rtype: bool
    """
    if p == path_prefix or p.startswith(path_prefix + "/"):
      print("Removing %s from %s." % (p, env_var))
      return False
    return True
  ps = filter(f, ps)
  os.environ[env_var] = ":".join(ps)


def get_login_username():
  """
  :rtype: str
  :return: the username of the current user.
  Use this as a replacement for os.getlogin().
  """
  import os
  if sys.platform == 'win32':
    return os.getlogin()
  import pwd
  try:
    return pwd.getpwuid(os.getuid())[0]
  except KeyError:
    # pwd.getpwuid() can throw KeyError: 'getpwuid(): uid not found: 12345'
    # this can happen e.g. in a docker environment with mapped uids unknown to the docker OS
    return str(os.getuid())


def get_temp_dir():
  """
  :rtype: str
  :return: e.g. "/tmp/$USERNAME"
  """
  username = get_login_username()
  for envname in ['TMPDIR', 'TEMP', 'TMP']:
    dirname = os.getenv(envname)
    if dirname:
      return "%s/%s" % (dirname, username)
  return "/tmp/%s" % username


def get_cache_dir():
  """
  :return: used to cache non-critical things. by default get_temp_dir. unless you define env RETURNN_CACHE_DIR
  :rtype: str
  """
  if "RETURNN_CACHE_DIR" in os.environ:
    return os.environ["RETURNN_CACHE_DIR"]
  return get_temp_dir()


class LockFile(object):
  """
  Simple lock file.
  """

  def __init__(self, directory, name="lock_file", lock_timeout=1 * 60 * 60):
    """
    :param str directory:
    :param int|float lock_timeout: in seconds
    """
    self.directory = directory
    self.name = name
    self.fd = None
    self.lock_timeout = lock_timeout
    self.lockfile = "%s/%s" % (directory, name)

  def is_old_lockfile(self):
    """
    :return: Whether there is an existing lock file and the existing lock file is old.
    :rtype: bool
    """
    try:
      mtime = os.path.getmtime(self.lockfile)
    except OSError:
      mtime = None
    if mtime and (abs(time.time() - mtime) > self.lock_timeout):
      return True
    return False

  def maybe_remove_old_lockfile(self):
    """
    Removes an existing old lockfile, if there is one.
    """
    if not self.is_old_lockfile():
      return
    print("Removing old lockfile %r (probably crashed proc)." % self.lockfile)
    try:
      os.remove(self.lockfile)
    except OSError as exc:
      print("Remove lockfile exception %r. Ignoring it." % exc)

  def is_locked(self):
    """
    :return: whether there is an active (not old) lockfile
    :rtype: bool
    """
    if self.is_old_lockfile():
      return False
    try:
      return os.path.exists(self.lockfile)
    except OSError:
      return False

  def lock(self):
    """
    Acquires the lock.
    """
    import time
    import errno
    wait_count = 0
    while True:
      # Try to create directory if it does not exist.
      try:
        os.makedirs(self.directory)
      except OSError:
        pass  # Ignore any errors.
      # Now try to create the lock.
      try:
        self.fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        return
      except OSError as exc:
        # Possible errors:
        # ENOENT (No such file or directory), e.g. if the directory was deleted.
        # EEXIST (File exists), if the lock already exists.
        if exc.errno not in [errno.ENOENT, errno.EEXIST]:
          raise  # Other error, so reraise.
      # We did not get the lock.
      # Check if it is a really old one.
      self.maybe_remove_old_lockfile()
      # Wait a bit, and then retry.
      time.sleep(1)
      wait_count += 1
      if wait_count == 10:
        print("Waiting for lock-file: %s" % self.lockfile)

  def unlock(self):
    """
    Releases the lock.
    """
    os.close(self.fd)
    os.remove(self.lockfile)

  def __enter__(self):
    self.lock()

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.unlock()


def str_is_number(s):
  """
  :param str s: e.g. "1", ".3" or "x"
  :return: whether s can be casted to float or int
  :rtype: bool
  """
  try:
    float(s)
    return True
  except ValueError:
    return False


def sorted_values_from_dict(d):
  """
  :param dict[T,V] d:
  :rtype: list[V]
  """
  assert isinstance(d, dict)
  return [v for (k, v) in sorted(d.items())]


def dict_zip(keys, values):
  """
  :param list[T] keys:
  :param list[V] values:
  :rtype: dict[T,V]
  """
  assert len(keys) == len(values)
  return dict(zip(keys, values))


def parse_ld_conf_file(fn):
  """
  Via https://github.com/albertz/system-tools/blob/master/bin/find-lib-in-path.py.

  :param str fn: e.g. "/etc/ld.so.conf"
  :return: list of paths for libs
  :rtype: list[str]
  """
  from glob import glob
  paths = []
  for line in open(fn).read().splitlines():
    line = line.strip()
    if not line:
      continue
    if line.startswith("#"):
      continue
    if line.startswith("include "):
      for sub_fn in glob(line[len("include "):]):
        paths.extend(parse_ld_conf_file(sub_fn))
      continue
    paths.append(line)
  return paths


def get_ld_paths():
  """
  To be very correct, see man-page of ld.so.
  And here: http://unix.stackexchange.com/questions/354295/what-is-the-default-value-of-ld-library-path/354296
  Short version, not specific to an executable, in this order:
  - LD_LIBRARY_PATH
  - /etc/ld.so.cache (instead we will parse /etc/ld.so.conf)
  - /lib, /usr/lib (or maybe /lib64, /usr/lib64)
  Via https://github.com/albertz/system-tools/blob/master/bin/find-lib-in-path.py.

  :rtype: list[str]
  :return: list of paths to search for libs (*.so files)
  """
  paths = []
  if "LD_LIBRARY_PATH" in os.environ:
    paths.extend(os.environ["LD_LIBRARY_PATH"].split(":"))
  if os.path.exists("/etc/ld.so.conf"):
    paths.extend(parse_ld_conf_file("/etc/ld.so.conf"))
  paths.extend(["/lib", "/usr/lib", "/lib64", "/usr/lib64"])
  return paths


def find_lib(lib_name):
  """
  :param str lib_name: without postfix/prefix, e.g. "cudart" or "blas"
  :return: returns full path to lib or None
  :rtype: str|None
  """
  if sys.platform == "darwin":
    prefix = "lib"
    postfix = ".dylib"
  elif sys.platform == "win32":
    prefix = ""
    postfix = ".dll"
  else:
    prefix = "lib"
    postfix = ".so"
  for path in get_ld_paths():
    fn = "%s/%s%s%s" % (path, prefix, lib_name, postfix)
    if os.path.exists(fn):
      return fn
  return None


def read_sge_num_procs(job_id=None):
  """
  From the Sun Grid Engine (SGE), reads the num_proc setting for a particular job.
  If job_id is not provided and the JOB_ID env is set, it will use that instead (i.e. it uses the current job).
  This calls qstat to figure out this setting. There are multiple ways this can go wrong,
  so better catch any exception.

  :param int|None job_id:
  :return: num_proc
  :rtype: int|None
  """
  if not job_id:
    if not os.environ.get("SGE_ROOT"):
      return None
    try:
      # qint.py might overwrite JOB_ID but sets SGE_JOB_ID instead.
      job_id = int(os.environ.get("SGE_JOB_ID") or os.environ.get("JOB_ID") or 0)
    except ValueError as exc:
      raise Exception("read_sge_num_procs: %r, invalid JOB_ID: %r" % (exc, os.environ.get("JOB_ID")))
    if not job_id:
      return None
  from subprocess import Popen, PIPE, CalledProcessError
  sge_cmd = ["qstat", "-j", str(job_id)]
  proc = Popen(sge_cmd, stdout=PIPE)
  stdout, _ = proc.communicate()
  if proc.returncode:
    raise CalledProcessError(proc.returncode, sge_cmd, stdout)
  stdout = stdout.decode("utf8")
  ls = [
    line[len("hard resource_list:"):].strip()
    for line in stdout.splitlines() if line.startswith("hard resource_list:")]
  assert len(ls) == 1
  opts = dict([opt.split("=", 1) for opt in ls[0].split(",")])  # noqa
  try:
    return int(opts["num_proc"])
  except ValueError as exc:
    raise Exception("read_sge_num_procs: %r, invalid num_proc %r for job id %i.\nline: %r" % (
      exc, opts["num_proc"], job_id, ls[0]))


def get_number_available_cpus():
  """
  :return: number of available CPUs, if we can figure it out
  :rtype: int|None
  """
  if hasattr(os, "sched_getaffinity"):  # Python >=3.4
    return len(os.sched_getaffinity(0))
  try:
    # noinspection PyPackageRequirements,PyUnresolvedReferences
    import psutil
    proc = psutil.Process()
    if hasattr(proc, "cpu_affinity"):
      return len(proc.cpu_affinity())
  except ImportError:
    pass
  if hasattr(os, "sysconf") and "SC_NPROCESSORS_ONLN" in os.sysconf_names:
    return os.sysconf("SC_NPROCESSORS_ONLN")
  if hasattr(os, "cpu_count"):  # Python >=3.4
    return os.cpu_count()  # not quite correct; that are all in the system
  return None


def guess_requested_max_num_threads(log_file=None, fallback_num_cpus=True):
  """
  :param io.File log_file:
  :param bool fallback_num_cpus:
  :rtype: int|None
  """
  try:
    sge_num_procs = read_sge_num_procs()
  except Exception as exc:
    if log_file:
      print("Error while getting SGE num_proc: %r" % exc, file=log_file)
  else:
    if sge_num_procs:
      if log_file:
        print("Use num_threads=%i (but min 2) via SGE num_proc." % sge_num_procs, file=log_file)
      return max(sge_num_procs, 2)
  omp_num_threads = int(os.environ.get("OMP_NUM_THREADS") or 0)
  if omp_num_threads:
    # Minimum of 2 threads, should not hurt.
    if log_file:
      print("Use num_threads=%i (but min 2) via OMP_NUM_THREADS." % omp_num_threads, file=log_file)
    return max(omp_num_threads, 2)
  if fallback_num_cpus:
    return get_number_available_cpus()
  return None


TheanoFlags = {
  key: value for (key, value) in [s.split("=", 1) for s in os.environ.get("THEANO_FLAGS", "").split(",") if s]}


def _consider_check_for_gpu():
  """
  There are cases where nvidia-smi could hang.
  (Any read of /proc/modules might hang in that case, maybe caused
   by trying to `modprobe nvidia` to check if there is a Nvidia card.)
  This sometimes happens in our SGE cluster on nodes without Nvidia cards.
  Maybe it's also a Linux Kernel bug.
  Anyway, just avoid any such check if we don't asked for a GPU.

  :rtype: bool
  """
  if "device" in TheanoFlags:
    dev = TheanoFlags["device"]
    if dev.startswith("gpu") or dev.startswith("cuda"):
      return True
    # THEANO_FLAGS will overwrite this config option. See rnn.initDevices().
    return False
  # noinspection PyBroadException
  try:
    from returnn.config import get_global_config
    config = get_global_config()
  except Exception:
    config = None
  if config:
    for dev in config.list('device', []):
      if dev.startswith("gpu") or dev.startswith("cuda"):
        return True
      if dev == "all":
        return True
  return False


def get_gpu_names():
  """
  :rtype: list[str]
  """
  if not _consider_check_for_gpu():
    return []
  if os.name == 'nt':
    return "GeForce GTX 770"  # TODO
  elif sys.platform == 'darwin':
    # TODO parse via xml output
    return sys_cmd_out_lines(
      "system_profiler SPDisplaysDataType | "
      "grep 'Chipset Model: NVIDIA' | "
      "sed 's/.*Chipset Model: NVIDIA *//;s/ *$//'")
  else:
    try:
      return sys_cmd_out_lines('nvidia-smi -L | cut -d \'(\' -f 1 | cut -d \' \' -f 3- | sed -e \'s/\\ $//\'')
    except CalledProcessError:
      return []


def _get_num_gpu_devices():
  """
  :return: cpu,gpu
  :rtype: (int,int)
  """
  if os.name == 'nt':
    return 1, 1  # TODO
  elif sys.platform == 'darwin':
    return (
      int(sys_cmd_out_lines("sysctl -a | grep machdep.cpu.core_count | awk '{print $2}'")[0]),
      len(sys_cmd_out_lines("system_profiler SPDisplaysDataType | grep 'Chipset Model: NVIDIA' | cat")))
  else:
    num_cpus = len(sys_cmd_out_lines('cat /proc/cpuinfo | grep processor')) or 1
    num_gpus = 0
    if _consider_check_for_gpu():
      try:
        num_gpus = len(sys_cmd_out_lines('nvidia-smi -L'))
      except CalledProcessError:
        pass
    return num_cpus, num_gpus


_num_devices = None


def get_num_gpu_devices():
  """
  :return: (cpu count, gpu count)
  :rtype: (int, int)
  """
  global _num_devices
  if _num_devices is not None:
    return _num_devices
  _num_devices = _get_num_gpu_devices()
  return _num_devices


def have_gpu():
  """
  :rtype: bool
  """
  cpus, gpus = get_num_gpu_devices()
  return gpus > 0


def try_and_ignore_exception(f):
  """
  Calls ``f``, and ignores any exception.

  :param ()->T f:
  :return: whatever ``f`` returns, or None
  :rtype: T|None
  """
  try:
    return f()
  except Exception as exc:
    print("try_and_ignore_exception: %r failed: %s" % (f, exc))
    sys.excepthook(*sys.exc_info())
    return None


def try_get_stack_frame(depth=1):
  """
  :param int depth:
  :rtype: types.FrameType|None
  :return: caller function name. this is just for debugging
  """
  # noinspection PyBroadException
  try:
    # noinspection PyProtectedMember,PyUnresolvedReferences
    frame = sys._getframe(depth + 1)  # one more to count ourselves
    return frame
  except Exception:
    return None


def try_get_caller_name(depth=1, fallback=None):
  """
  :param int depth:
  :param str|None fallback: this is returned if we fail for some reason
  :rtype: str|None
  :return: caller function name. this is just for debugging
  """
  frame = try_get_stack_frame(depth + 1)  # one more to count ourselves
  if frame:
    from .better_exchook import get_func_str_from_code_object
    return get_func_str_from_code_object(frame.f_code)
  return fallback


def camel_case_to_snake_case(name):
  """
  :param str name: e.g. "CamelCase"
  :return: e.g. "camel_case"
  :rtype: str
  """
  s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def get_hostname():
  """
  :return: e.g. "cluster-cn-211"
  :rtype: str
  """
  # check_output(["hostname"]).strip().decode("utf8")
  import socket
  return socket.gethostname()


def is_running_on_cluster():
  """
  :return: i6 specific. Whether we run on some of the cluster nodes.
  :rtype: bool
  """
  return get_hostname().startswith("cluster-cn-")


start_time = time.time()


def get_utc_start_time_filename_part():
  """
  :return: string which can be used as part of a filename, which represents the start time of RETURNN in UTC
  :rtype: str
  """
  return time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime(start_time))


def maybe_make_dirs(dirname):
  """
  Creates the directory if it does not yet exist.

  :param str dirname: The path of the directory
  """
  import os
  if not os.path.exists(dirname):
    try:
      os.makedirs(dirname)
    except Exception as exc:
      print("maybe_create_folder: exception creating dir:", exc)
      # Maybe a concurrent process, e.g. tf.compat.v1.summary.FileWriter created it in the mean-while,
      # so then it would be ok now if it exists, but fail if it does not exist.
      assert os.path.exists(dirname)


def log_runtime_info_to_dir(path, config):
  """
  This will write multiple logging information into the path.
  It will create returnn.*.log with some meta information,
  as well as copy the used config file.

  :param str path: directory path
  :param Config.Config config:
  """
  import os
  import sys
  import shutil
  from returnn.config import Config
  try:
    hostname = get_hostname()
    content = [
      "Time: %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)),
      "Call: %s" % (sys.argv,),
      "Path: %s" % (os.getcwd(),),
      "Hostname: %s" % get_hostname(),
      "PID: %i" % os.getpid(),
      "Returnn: %s" % (describe_returnn_version(),),
      "TensorFlow: %s" % (describe_tensorflow_version(),),
      "Config files: %s" % (config.files,),
    ]
    maybe_make_dirs(path)
    log_fn = "%s/returnn.%s.%s.%i.log" % (path, get_utc_start_time_filename_part(), hostname, os.getpid())
    if not os.path.exists(log_fn):
      with open(log_fn, "w") as f:
        f.write(
          "Returnn log file:\n" +
          "".join(["%s\n" % s for s in content]) +
          "\n")
    for fn in config.files:
      base_fn = os.path.basename(fn)
      target_fn = "%s/%s" % (path, base_fn)
      if os.path.exists(target_fn):
        continue
      shutil.copy(fn, target_fn)
      config_type = Config.get_config_file_type(fn)
      comment_prefix = "#"
      if config_type == "js":
        comment_prefix = "//"
      with open(target_fn, "a") as f:
        f.write(
          "\n\n\n" +
          "".join(
            ["%s Config-file copied for logging purpose by Returnn.\n" % comment_prefix] +
            ["%s %s\n" % (comment_prefix, s) for s in content]) +
          "\n")
  except OSError as exc:
    if "Disk quota" in str(exc):
      print("log_runtime_info_to_dir: Error, cannot write: %s" % exc)
    else:
      raise


def should_write_to_disk(config):
  """
  :param Config.Config config:
  :rtype: bool
  """
  if config.is_true("use_horovod"):
    # noinspection PyPackageRequirements,PyUnresolvedReferences
    import horovod.tensorflow as hvd
    if hvd.rank() != 0:
      return False
  if config.is_true("dry_run"):
    return False
  return True


class NativeCodeCompiler(object):
  """
  Helper class to compile native C/C++ code on-the-fly.
  """

  CacheDirName = "returnn_native"
  CollectedCompilers = None  # type: typing.Optional[typing.List[NativeCodeCompiler]]

  def __init__(self, base_name, code_version, code,
               is_cpp=True, c_macro_defines=None, ld_flags=None,
               include_paths=(), include_deps=None,
               static_version_name=None, should_cleanup_old_all=True, should_cleanup_old_mydir=False,
               use_cxx11_abi=False, log_stream=None, verbose=False):
    """
    :param str base_name: base name for the module, e.g. "zero_out"
    :param int|tuple[int] code_version: check for the cache whether to reuse
    :param str code: the source code itself
    :param bool is_cpp: if False, C is assumed
    :param dict[str,str|int]|None c_macro_defines: e.g. {"TENSORFLOW": 1}
    :param list[str]|None ld_flags: e.g. ["-lblas"]
    :param list[str]|tuple[str] include_paths:
    :param list[str]|None include_deps: if provided and an existing lib file, we will check if any dependency is newer
      and we need to recompile. we could also do it automatically via -MD but that seems overkill and too slow.
    :param str|None static_version_name: normally, we use .../base_name/hash as the dir
      but this would use .../base_name/static_version_name.
    :param bool should_cleanup_old_all: whether we should look in the cache dir
      and check all ops if we can delete some old ones which are older than some limit (self._cleanup_time_limit_days)
    :param bool should_cleanup_old_mydir: whether we should delete our op dir before we compile there.
    :param typing.TextIO|None log_stream: file stream for print statements
    :param bool verbose: be slightly more verbose
    """
    if self.CollectedCompilers is not None:
      self.CollectedCompilers.append(self)
    self.verbose = verbose
    self.cache_dir = "%s/%s" % (get_cache_dir(), self.CacheDirName)
    self._include_paths = list(include_paths)
    self.base_name = base_name
    self.code_version = code_version
    self.code = code
    self.is_cpp = is_cpp
    self.c_macro_defines = c_macro_defines or {}
    self.ld_flags = ld_flags or []
    self.include_deps = include_deps
    self.static_version_name = static_version_name
    self._code_hash = self._make_code_hash()
    self._info_dict = self._make_info_dict()
    self._hash = self._make_hash()
    self._ctypes_lib = None
    if should_cleanup_old_all:
      self._cleanup_old()
    self._should_cleanup_old_mydir = should_cleanup_old_mydir
    self.use_cxx11_abi = use_cxx11_abi
    self._log_stream = log_stream
    if self.verbose:
      print("%s: %r" % (self.__class__.__name__, self), file=log_stream)

  def __repr__(self):
    return "<%s %r in %r>" % (self.__class__.__name__, self.base_name, self._mod_path)

  @property
  def _mod_path(self):
    return "%s/%s/%s" % (self.cache_dir, self.base_name, self.static_version_name or self._hash[:10])

  @property
  def _info_filename(self):
    return "%s/info.py" % (self._mod_path,)

  @property
  def _so_filename(self):
    return "%s/%s.so" % (self._mod_path, self.base_name)

  @property
  def _c_filename(self):
    if self.is_cpp:
      return "%s/%s.cc" % (self._mod_path, self.base_name)
    return "%s/%s.c" % (self._mod_path, self.base_name)

  _cleanup_time_limit_days = 60

  def _cleanup_old(self):
    mod_path = self._mod_path  # .../base_name/hash
    base_mod_path = os.path.dirname(mod_path)  # .../base_name
    my_mod_path_name = os.path.basename(mod_path)
    if not os.path.exists(base_mod_path):
      return
    import time
    cleanup_time_limit_secs = self._cleanup_time_limit_days * 24 * 60 * 60
    for p in os.listdir(base_mod_path):
      if p == my_mod_path_name:
        continue
      full_dir_path = "%s/%s" % (base_mod_path, p)
      if not os.path.isdir(full_dir_path):
        continue  # ignore for now
      lock = LockFile(full_dir_path)
      if lock.is_locked():
        continue
      lock.maybe_remove_old_lockfile()
      info_path = "%s/info.py" % full_dir_path
      if not os.path.exists(info_path):
        self._cleanup_old_path(full_dir_path, reason="corrupt dir, missing info.py")
        continue
      so_path = "%s/%s.so" % (full_dir_path, self.base_name)
      if not os.path.exists(so_path):
        self._cleanup_old_path(full_dir_path, reason="corrupt dir, missing so")
        continue
      dt = time.time() - os.path.getmtime(so_path)
      if dt > cleanup_time_limit_secs:
        self._cleanup_old_path(full_dir_path, reason="%s old" % hms(dt))

  def _cleanup_old_path(self, p, reason):
    print("%s delete old, %s: %s" % (self.__class__.__name__, reason, p))
    assert os.path.exists(p)
    import shutil
    try:
      shutil.rmtree(p)
    except OSError as exc:
      print("%s delete exception (%s). Will ignore and try to continue anyway." % (self.__class__.__name__, exc))

  def _load_info(self):
    """
    :rtype: dict[str]|None
    """
    filename = self._info_filename
    if not os.path.exists(filename):
      return None
    s = open(filename).read()
    res = eval(s)
    assert isinstance(res, dict)
    return res

  _relevant_info_keys = ("code_version", "code_hash", "c_macro_defines", "ld_flags", "compiler_bin")

  def _make_info_dict(self):
    """
    :rtype: dict[str]
    """
    return {
      "base_name": self.base_name,
      "include_paths": self._include_paths,
      "code_version": self.code_version,
      "code_hash": self._code_hash,
      "c_macro_defines": self.c_macro_defines,
      "ld_flags": self.ld_flags,
      "compiler_bin": self._get_compiler_bin(),
    }

  def _make_code_hash(self):
    import hashlib
    h = hashlib.md5()
    h.update(self.code.encode("utf8"))
    return h.hexdigest()

  def _make_hash(self):
    import hashlib
    h = hashlib.md5()
    h.update("{".encode("utf8"))
    for key in self._relevant_info_keys:
      h.update(("%s:{%s}" % (key, self._info_dict[key])).encode("utf8"))
    h.update("}".encode("utf8"))
    return h.hexdigest()

  def _save_info(self):
    filename = self._info_filename
    with open(filename, "w") as f:
      f.write("%s\n" % better_repr(self._info_dict))

  def _need_recompile(self):
    """
    :rtype: bool
    """
    if not os.path.exists(self._so_filename):
      return True
    if self.include_deps:
      so_mtime = os.path.getmtime(self._so_filename)
      for fn in self.include_deps:
        if os.path.getmtime(fn) > so_mtime:
          return True
    old_info = self._load_info()
    new_info = self._make_info_dict()
    if not old_info:
      return True
    # The hash already matched but very unlikely, this could be a collision.
    # Anyway, just do this very cheap check.
    for key in self._relevant_info_keys:
      if key not in old_info:
        return True
      if old_info[key] != new_info[key]:
        return True
    # If no code version is provided, we could also check the code itself now.
    # But I think this is overkill.
    return False

  def _maybe_compile(self):
    """
    On successful return, self._so_filename should exist and be up-to-date.
    """
    if not self._need_recompile():
      if self.verbose:
        print("%s: No need to recompile: %s" % (self.__class__.__name__, self._so_filename))
      # Touch it so that we can see that we used it recently.
      os.utime(self._info_filename, None)
      return
    lock = LockFile(self._mod_path)
    if not self._need_recompile():  # check again
      if self.verbose:
        print("%s: No need to recompile after we waited: %s" % (self.__class__.__name__, self._so_filename))
      os.utime(self._info_filename, None)
      return
    if self._should_cleanup_old_mydir and not lock.is_locked():
      if os.path.exists(self._mod_path):
        self._cleanup_old_path(self._mod_path, reason="need recompile")
    with lock:
      self._maybe_compile_inner()

  def _get_compiler_bin(self):
    """
    :rtype: str
    """
    if self.is_cpp:
      return "g++"
    return "gcc"

  def _transform_compiler_opts(self, opts):
    """
    :param list[str] opts:
    :rtype: list[str]
    """
    return opts

  @classmethod
  def _transform_ld_flag(cls, opt):
    """
    :param str opt:
    :rtype: str
    """
    if sys.platform == "darwin":
      # It seems some versions of MacOS ld cannot handle the `-l:filename` argument correctly.
      # E.g. TensorFlow 1.14 incorrectly uses this.
      # https://github.com/tensorflow/tensorflow/issues/30564
      if opt.startswith("-l:lib") and opt.endswith(".dylib"):
        return "-l%s" % opt[len("-l:lib"):-len(".dylib")]
    return opt

  def _maybe_compile_inner(self):
    # Directory should be created by the locking mechanism.
    assert os.path.exists(self._mod_path)
    with open(self._c_filename, "w") as f:
      f.write(self.code)
    common_opts = ["-shared", "-O2"]
    if self.is_cpp:
      common_opts += ["-std=c++11"]
    if sys.platform == "darwin":
      common_opts += ["-undefined", "dynamic_lookup"]
    for include_path in self._include_paths:
      common_opts += ["-I", include_path]
    compiler_opts = ["-fPIC", "-v"]
    common_opts += self._transform_compiler_opts(compiler_opts)
    common_opts += ["-D_GLIBCXX_USE_CXX11_ABI=%i" % (1 if self.use_cxx11_abi else 0)]
    common_opts += ["-D%s=%s" % item for item in sorted(self.c_macro_defines.items())]
    common_opts += ["-g"]
    opts = common_opts + [self._c_filename, "-o", self._so_filename]
    opts += list(map(self._transform_ld_flag, self.ld_flags))
    cmd_bin = self._get_compiler_bin()
    cmd_args = [cmd_bin] + opts
    from subprocess import Popen, PIPE, STDOUT, CalledProcessError
    print("%s call: %s" % (self.__class__.__name__, " ".join(cmd_args)), file=self._log_stream)
    proc = Popen(cmd_args, cwd=self._mod_path, stdout=PIPE, stderr=STDOUT)
    stdout, stderr = proc.communicate()
    assert stderr is None  # should only have stdout
    if proc.returncode != 0:
      print("%s: %s failed." % (self.__class__.__name__, cmd_bin))
      print("Original stdout/stderr:")
      print(stdout.decode("utf8"))
      print()
      if cmd_bin.endswith("/nvcc") and b"error: constexpr function return is non-constant" in stdout:
        print("This might be the error: https://github.com/tensorflow/tensorflow/issues/22766")
        print()
      if cmd_bin.endswith("/nvcc") and b"gcc versions later than" in stdout:
        print("Your GCC version might be too new. This is a problem with some nvcc versions.")
        print()
      raise CalledProcessError(returncode=proc.returncode, cmd=cmd_args)
    assert os.path.exists(self._so_filename)
    with open("%s/compile.log" % self._mod_path, "wb") as f:
      if self.verbose:
        print("%s: write compile log to: %s" % (self.__class__.__name__, f.name))
      f.write(("+ %s\n" % " ".join(cmd_args)).encode("utf8"))
      f.write(stdout)
    self._save_info()
    assert not self._need_recompile()

  def load_lib_ctypes(self):
    """
    :rtype: ctypes.CDLL
    """
    if self._ctypes_lib:
      return self._ctypes_lib
    self._maybe_compile()
    import ctypes
    self._ctypes_lib = ctypes.cdll.LoadLibrary(self._so_filename)
    return self._ctypes_lib

  def get_lib_filename(self):
    """
    :rtype: str
    """
    self._maybe_compile()
    return self._so_filename


# See :func:`maybe_restart_returnn_with_atfork_patch` below for why you might want to use this.
_c_code_patch_atfork = """
#define _GNU_SOURCE
#include <sched.h>
#include <signal.h>
#include <sys/syscall.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// https://stackoverflow.com/questions/46845496/ld-preload-and-linkage
// https://stackoverflow.com/questions/46810597/forkexec-without-atfork-handlers

int pthread_atfork(void (*prepare)(void), void (*parent)(void), void (*child)(void)) {
  printf("Ignoring pthread_atfork call!\\n");
  fflush(stdout);
  return 0;
}

int __register_atfork(void (*prepare)(void), void (*parent)(void), void (*child)(void)) {
  printf("Ignoring __register_atfork call!\\n");
  fflush(stdout);
  return 0;
}

// Another way to ignore atfork handlers: Override fork.
#ifdef __linux__ // only works on Linux currently
pid_t fork(void) {
  return syscall(SYS_clone, SIGCHLD, 0);
}
#endif

__attribute__((constructor))
void patch_atfork_init() {
  setenv("__RETURNN_ATFORK_PATCHED", "1", 1);
}
"""


def get_patch_atfork_lib():
  """
  :return: path to our patch_atfork lib. see :func:`maybe_restart_returnn_with_atfork_patch`
  :rtype: str
  """
  native = NativeCodeCompiler(
    base_name="patch_atfork", code_version=2, code=_c_code_patch_atfork, is_cpp=False)
  fn = native.get_lib_filename()
  return fn


def maybe_restart_returnn_with_atfork_patch():
  """
  What we want: subprocess.Popen to always work.
  Problem: It uses fork+exec internally in subprocess_fork_exec, via _posixsubprocess.fork_exec.
  That is a problem because fork can trigger any atfork handlers registered via pthread_atfork,
  and those can crash/deadlock in some cases.

  https://github.com/tensorflow/tensorflow/issues/13802
  https://github.com/xianyi/OpenBLAS/issues/240
  https://trac.sagemath.org/ticket/22021
  https://bugs.python.org/issue31814
  https://stackoverflow.com/questions/46845496/ld-preload-and-linkage
  https://stackoverflow.com/questions/46810597/forkexec-without-atfork-handlers

  The solution here: Just override pthread_atfork, via LD_PRELOAD.
  Note that in some cases, this is not enough (see the SO discussion),
  so we also overwrite fork itself.
  See also tests/test_fork_exec.py for a demo.
  """
  if os.environ.get("__RETURNN_ATFORK_PATCHED") == "1":
    print("Running with patched atfork.")
    return
  if os.environ.get("__RETURNN_TRY_ATFORK_PATCHED") == "1":
    print("Patching atfork did not work! Will continue anyway.")
    return
  lib = get_patch_atfork_lib()
  env = os.environ.copy()
  env["DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD"] = lib
  env["__RETURNN_TRY_ATFORK_PATCHED"] = "1"
  print("Restarting Returnn with atfork patch...", sys.executable, sys.argv)
  sys.stdout.flush()
  os.execvpe(sys.executable, [sys.executable] + sys.argv, env)
  print("execvpe did not work?")


class Stats:
  """
  Collects mean and variance, running average.

  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
  """

  def __init__(self, format_str=None):
    """
    :param None|((float|numpy.ndarray)->str) format_str:
    """
    self.format_str = format_str or str
    self.mean = 0.0
    self.mean_sq = 0.0
    self.var = 0.0
    self.min = None
    self.max = None
    self.total_data_len = 0
    self.num_seqs = 0

  def __str__(self):
    if self.num_seqs > 0:
      if self.num_seqs == self.total_data_len:
        extra_str = "avg_data_len=1"
      else:
        extra_str = "total_data_len=%i, avg_data_len=%f" % (
          self.total_data_len, float(self.total_data_len) / self.num_seqs)
      return "Stats(mean=%s, std_dev=%s, min=%s, max=%s, num_seqs=%i, %s)" % (
        self.format_str(self.get_mean()), self.format_str(self.get_std_dev()),
        self.format_str(self.min), self.format_str(self.max), self.num_seqs, extra_str)
    return "Stats(num_seqs=0)"

  def collect(self, data):
    """
    :param numpy.ndarray|list[int]|list[float] data: shape (time, dim) or (time,)
    """
    import numpy
    if isinstance(data, (list, tuple)):
      data = numpy.array(data)
    assert isinstance(data, numpy.ndarray)
    assert data.ndim >= 1
    if data.shape[0] == 0:
      return
    self.num_seqs += 1
    data_min = numpy.min(data, axis=0)
    data_max = numpy.max(data, axis=0)
    if self.min is None:
      self.min = data_min
      self.max = data_max
    else:
      self.min = numpy.minimum(self.min, data_min)
      self.max = numpy.maximum(self.max, data_max)
    new_total_data_len = self.total_data_len + data.shape[0]
    mean_diff = numpy.mean(data, axis=0) - self.mean
    m_a = self.var * self.total_data_len
    m_b = numpy.var(data, axis=0) * data.shape[0]
    m2 = m_a + m_b + mean_diff ** 2 * self.total_data_len * data.shape[0] / new_total_data_len
    self.var = m2 / new_total_data_len
    data_sum = numpy.sum(data, axis=0)
    delta = data_sum - self.mean * data.shape[0]
    self.mean += delta / new_total_data_len
    delta_sq = numpy.sum(data * data, axis=0) - self.mean_sq * data.shape[0]
    self.mean_sq += delta_sq / new_total_data_len
    self.total_data_len = new_total_data_len

  def get_mean(self):
    """
    :return: mean, shape (dim,)
    :rtype: numpy.ndarray
    """
    assert self.num_seqs > 0
    return self.mean

  def get_std_dev(self):
    """
    :return: std dev, shape (dim,)
    :rtype: numpy.ndarray
    """
    import numpy
    assert self.num_seqs > 0
    return numpy.sqrt(self.var)
    # return numpy.sqrt(self.mean_sq - self.mean * self.mean)

  def dump(self, output_file_prefix=None, stream=None, stream_prefix=""):
    """
    :param str|None output_file_prefix: if given, will numpy.savetxt mean|std_dev to disk
    :param str stream_prefix:
    :param io.TextIOBase stream: sys.stdout by default
    """
    if stream is None:
      stream = sys.stdout
    import numpy
    print("%sStats:" % stream_prefix, file=stream)
    if self.num_seqs != self.total_data_len:
      print("  %i seqs, %i total frames, %f average frames" % (
        self.num_seqs, self.total_data_len, self.total_data_len / float(self.num_seqs)), file=stream)
    else:
      print("  %i seqs" % (self.num_seqs,), file=stream)
    print("  Mean: %s" % (self.format_str(self.get_mean()),), file=stream)
    print("  Std dev: %s" % (self.format_str(self.get_std_dev()),), file=stream)
    print("  Min/max: %s / %s" % (self.format_str(self.min), self.format_str(self.max)), file=stream)
    # print("Std dev (naive): %s" % numpy.sqrt(self.mean_sq - self.mean * self.mean), file=stream)
    if output_file_prefix:
      print("  Write mean/std-dev to %s.(mean|std_dev).txt." % (output_file_prefix,), file=stream)
      numpy.savetxt("%s.mean.txt" % output_file_prefix, self.get_mean())
      numpy.savetxt("%s.std_dev.txt" % output_file_prefix, self.get_std_dev())


def is_namedtuple(cls):
  """
  :param T cls: tuple, list or namedtuple type
  :return: whether cls is a namedtuple type
  :rtype: bool
  """
  return issubclass(cls, tuple) and cls is not tuple


def make_seq_of_type(cls, seq):
  """
  :param type[T] cls: e.g. tuple, list or namedtuple
  :param list|tuple|T seq:
  :return: cls(seq) or cls(*seq)
  :rtype: T|list|tuple
  """
  assert issubclass(cls, (list, tuple))
  if is_namedtuple(cls):
    return cls(*seq)  # noqa
  return cls(seq)  # noqa


@contextlib.contextmanager
def dummy_noop_ctx():
  """
  Provides a no-op context manager.
  """
  yield None


def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.
  Code adapted from Google Tensor2Tensor.

  Args:
    segment (list[int]|list[str]): text segment from which n-grams will be extracted.
    max_order (int): maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  import collections
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i + order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus,
                 translation_corpus,
                 max_order=4,
                 use_bp=True):
  """Computes BLEU score of translated segments against one or more references.
  Code adapted from Google Tensor2Tensor.

  Args:
    reference_corpus (list[list[int]|list[str]]): list of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus (list[list[int]|list[str]]): list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order (int): Maximum n-gram order to use when computing BLEU score.
    use_bp (bool): boolean, whether to apply brevity penalty.

  Returns:
    BLEU score.
  """
  import math
  reference_length = 0
  translation_length = 0
  bp = 1.0
  geo_mean = 0

  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order

  for (references, translations) in zip(reference_corpus, translation_corpus):
    reference_length += len(references)
    translation_length += len(translations)
    ref_ngram_counts = _get_ngrams(references, max_order)
    translation_ngram_counts = _get_ngrams(translations, max_order)

    overlap = {ngram: min(count, translation_ngram_counts[ngram])
               for ngram, count in ref_ngram_counts.items()}

    for ngram in overlap:
      matches_by_order[len(ngram) - 1] += overlap[ngram]
    for ngram in translation_ngram_counts:
      possible_matches_by_order[len(ngram) - 1] += translation_ngram_counts[ngram]

  precisions = [0.0] * max_order
  smooth = 1.0
  for i in range(0, max_order):
    if possible_matches_by_order[i] > 0:
      precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
      if matches_by_order[i] > 0:
        precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
      else:
        smooth *= 2
        precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
    else:
      precisions[i] = 0.0

  if max(precisions) > 0:
    p_log_sum = sum(math.log(p) for p in precisions if p)
    geo_mean = math.exp(p_log_sum / max_order)

  if use_bp:
    ratio = translation_length / reference_length
    if ratio < 1e-30:
      bp = 0.0
    elif ratio < 1.0:
      bp = math.exp(1 - 1. / ratio)
    else:
      bp = 1.0
  bleu = geo_mean * bp
  return np.float32(bleu)


# noinspection PyPackageRequirements
def monkeyfix_glib():
  """
  Fixes some stupid bugs such that SIGINT is not working.
  This is used by audioread, and indirectly by librosa for loading audio.
  https://stackoverflow.com/questions/16410852/
  See also :func:`monkeypatch_audioread`.
  """
  try:
    import gi
  except ImportError:
    return
  try:
    from gi.repository import GLib
  except ImportError:
    # noinspection PyUnresolvedReferences
    from gi.overrides import GLib
  # Do nothing.
  # The original behavior would install a SIGINT handler which calls GLib.MainLoop.quit(),
  # and then reraise a KeyboardInterrupt in that thread.
  # However, we want and expect to get the KeyboardInterrupt in the main thread.
  GLib.MainLoop.__init__ = lambda *args, **kwargs: None


def monkeypatch_audioread():
  """
  audioread does not behave optimal in some cases.
  E.g. each call to _ca_available() takes quite long because of the ctypes.util.find_library usage.
  We will patch this.

  However, the recommendation would be to not use audioread (librosa.load).
  audioread uses Gstreamer as a backend by default currently (on Linux).
  Gstreamer has multiple issues. See also :func:`monkeyfix_glib`, and here for discussion:
  https://github.com/beetbox/audioread/issues/62
  https://github.com/beetbox/audioread/issues/63

  Instead, use PySoundFile, which is also faster. See here for discussions:
  https://github.com/beetbox/audioread/issues/64
  https://github.com/librosa/librosa/issues/681
  """
  try:
    # noinspection PyPackageRequirements
    import audioread
  except ImportError:
    return
  # noinspection PyProtectedMember
  res = audioread._ca_available()
  audioread._ca_available = lambda: res


_cf_cache = {}
_cf_msg_printed = False


def cf(filename):
  """
  Cache manager. i6 specific.

  :return: filename
  :rtype: str
  """
  global _cf_msg_printed
  import os
  from subprocess import check_output
  if filename in _cf_cache:
    return _cf_cache[filename]
  debug_mode = int(os.environ.get("DEBUG", 0))
  if debug_mode or get_hostname() == "cluster-cn-211" or not is_running_on_cluster():
    if not _cf_msg_printed:
      print("Cache manager: not used, use local file: %s (discard further messages)" % filename)
      _cf_msg_printed = True
    return filename  # for debugging
  try:
    cached_fn = check_output(["cf", filename]).strip().decode("utf8")
  except CalledProcessError:
    if not _cf_msg_printed:
      print("Cache manager: Error occurred, using local file")
      _cf_msg_printed = True
    return filename
  assert os.path.exists(cached_fn)
  _cf_cache[filename] = cached_fn
  return cached_fn


def binary_search_any(cmp, low, high):
  """
  Binary search for a custom compare function.

  :param (int)->int cmp: e.g. cmp(idx) == compare(array[idx], key)
  :param int low: inclusive
  :param int high: exclusive
  :rtype: int|None
  """
  while low < high:
    mid = (low + high) // 2
    r = cmp(mid)
    if r < 0:
      low = mid + 1
    elif r > 0:
      high = mid
    else:
      return mid
  return low


def generic_import_module(filename):
  """
  :param str filename:
    We try to be clever about filename.
    If it looks like a module name, just do importlib.import_module.
    If it looks like a filename, search for a base path (which does not have __init__.py),
    add that path to sys.path if needed, and import the remaining where "/" is replaced by "."
    and the file extension is removed.
  :return: the module
  :rtype: types.ModuleType
  """
  assert filename
  import importlib
  if "/" not in filename:
    return importlib.import_module(filename)
  prefix_dir = ''
  if not os.path.exists(filename):
    assert filename[0] != '/'
    # Maybe relative to Returnn?
    prefix_dir = "%s/" % returnn_root_dir
  assert os.path.exists(prefix_dir + filename)
  assert filename.endswith('.py') or os.path.isdir(prefix_dir + filename)
  dirs = filename.split("/")
  dirs, base_fn = dirs[:-1], dirs[-1]
  assert len(dirs) >= 1
  for i in reversed(range(len(dirs))):
    d = prefix_dir + "/".join(dirs[:i + 1])
    assert os.path.isdir(d)
    if os.path.exists("%s/__init__.py" % d):
      continue
    if d not in sys.path:
      sys.path.append(d)
    m = ".".join(dirs[i + 1:] + [base_fn])
    if base_fn.endswith('.py'):
      m = m[:-3]
    return importlib.import_module(m)
  raise ValueError('cannot figure out base module path from %r' % filename)


def softmax(x, axis=None):
  """
  :param numpy.ndarray x:
  :param int|None axis:
  :rtype: numpy.ndarray
  """
  import numpy
  e_x = numpy.exp(x - numpy.max(x, axis=axis, keepdims=True))
  return e_x / numpy.sum(e_x, axis=axis, keepdims=True)


def collect_proc_maps_exec_files():
  """
  Currently only works on Linux...

  :return: list of mapped executables (libs)
  :rtype: list[str]
  """
  import re
  pid = os.getpid()
  fns = []
  for line in open("/proc/%i/maps" % pid, 'r').read().splitlines():  # for each mapped region
    # https://stackoverflow.com/questions/1401359/understanding-linux-proc-id-maps
    # address           perms offset  dev   inode   pathname
    # E.g.:
    # 7ff2de91c000-7ff2de91e000 rw-p 0017c000 08:02 794844                     /usr/lib/x86_64-linux-gnu/libstdc+...
    m = re.match(
      r'^([0-9A-Fa-f]+)-([0-9A-Fa-f]+)\s+([rwxps\-]+)\s+([0-9A-Fa-f]+)\s+([0-9A-Fa-f:]+)\s+([0-9]+)\s*(.*)$', line)
    assert m, "no match for %r" % line
    address_start, address_end, perms, offset, dev, i_node, path_name = m.groups()
    if "x" not in perms:
      continue
    if not path_name or path_name.startswith("[") or "(deleted)" in path_name:
      continue
    if path_name not in fns:
      fns.append(path_name)
  return fns


def find_sym_in_exec(fn, sym):
  """
  Uses ``objdump`` to list available symbols, and filters them by the given ``sym``.

  :param str fn: path
  :param str sym:
  :return: matched out, or None
  :rtype: str|None
  """
  from subprocess import CalledProcessError
  objdump = "objdump -T"
  if sys.platform == "darwin":
    objdump = "otool -IHGv"
  shell_cmd = "%s %s | grep %s" % (objdump, fn, sym)
  try:
    out = sys_exec_out(shell_cmd, shell=True)
  except CalledProcessError:  # none found
    return None
  assert isinstance(out, (str, unicode))
  out_lns = out.splitlines()
  out_lns = [ln for ln in out_lns if ".text" in ln]  # see objdump
  out_lns = [ln for ln in out_lns if sym in ln.split()]
  if not out_lns:
    return None
  return "Found %r in %r:\n%s" % (sym, fn, "\n".join(out_lns))


def dummy_numpy_gemm_call():
  """
  Just performs some GEMM call via Numpy.
  This makes sure that the BLAS library is loaded.
  """
  import numpy
  a = numpy.random.randn(5, 3).astype(numpy.float32)
  b = numpy.random.randn(3, 7).astype(numpy.float32)
  c = numpy.dot(a, b)
  assert numpy.isfinite(c).all()


_find_sgemm_lib_from_runtime_cached = None


def find_sgemm_libs_from_runtime():
  """
  Looks through all libs via :func:`collect_proc_maps_exec_files`,
  and searches for all which have the ``sgemm`` symbol.
  Currently only works on Linux (because collect_proc_maps_exec_files).

  :return: list of libs (their path)
  :rtype: list[str]
  """
  if not os.path.exists("/proc"):
    return None
  global _find_sgemm_lib_from_runtime_cached
  if _find_sgemm_lib_from_runtime_cached is not None:
    return _find_sgemm_lib_from_runtime_cached
  dummy_numpy_gemm_call()  # make sure that Numpy is loaded and Numpy sgemm is available
  fns = collect_proc_maps_exec_files()
  fns_with_sgemm = []
  for fn in fns:
    out = find_sym_in_exec(fn, "sgemm_")
    if out:
      fns_with_sgemm.append(fn)
  _find_sgemm_lib_from_runtime_cached = fns_with_sgemm
  return fns_with_sgemm


_find_libcudart_from_runtime_cached = None


def find_libcudart_from_runtime():
  """
  Looks through all libs via :func:`collect_proc_maps_exec_files`,
  and searches for all which have the ``sgemm`` symbol.
  Currently only works on Linux (because collect_proc_maps_exec_files).

  :return: list of libs (their path)
  :rtype: str|None
  """
  if not os.path.exists("/proc"):
    return None
  global _find_libcudart_from_runtime_cached
  if _find_libcudart_from_runtime_cached is not None:
    return _find_libcudart_from_runtime_cached[0]
  fns = collect_proc_maps_exec_files()
  for fn in fns:
    if re.match(".*/libcudart\\.so(\\..*)?", fn):
      _find_libcudart_from_runtime_cached = [fn]
      return fn
  _find_libcudart_from_runtime_cached = [None]
  return None
