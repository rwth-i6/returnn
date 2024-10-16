# -*- coding: utf8 -*-

"""
Various generic utilities, which are shared across different backend engines.
"""

from __future__ import annotations
from typing import Optional, Union, Any, Generic, TypeVar, Iterable, Tuple, Dict, List, Callable

import subprocess
from subprocess import CalledProcessError

import h5py
from collections import deque
import inspect
import os
import sys
import shlex
import math
import numpy
import numpy as np
import re
import time
import contextlib
import struct

try:
    import thread
except ImportError:
    import _thread as thread
import threading

from io import BytesIO
import typing
from returnn.log import log
import builtins

from .native_code_compiler import NativeCodeCompiler

PY3 = sys.version_info[0] >= 3

unicode = str
long = int
# noinspection PyShadowingBuiltins
input = builtins.input


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

returnn_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class NotSpecified:
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


class Entity:
    """
    This is a generic placeholder which can be used for enums or other identities.
    By intention it uses object.__eq__ and co, i.e. ``a == b`` iff ``a is b``.
    The name is just for debugging purpose.
    This is more efficient than using just the string directly in an enum.
    """

    def __init__(
        self, name: Optional[str] = None, *, global_base: Optional[Any] = None, global_name: Optional[str] = None
    ):
        """
        :param str|None name:
        """
        self.name = name
        if global_name and not global_base:
            frame = try_get_stack_frame(1)
            global_base = sys.modules[frame.f_globals["__name__"]]
        self.global_base = global_base
        self.global_name = global_name

    def __str__(self):
        return self.name or self.global_name or "<unnamed Entity>"

    def __repr__(self):
        return "<%s Entity>" % (self.name or self.global_name or "unnamed")

    def __reduce__(self):
        if self.global_name:
            # Sanity check that the global ref is correct.
            attrs = self.global_name.split(".")
            assert attr_chain(self.global_base, attrs) is self
            parent = attr_chain(self.global_base, attrs[:-1])
            assert getattr(parent, attrs[-1]) is self
            return getattr, (parent, attrs[-1])
        raise Exception("Cannot pickle Entity object. (%r)" % self)

    def __getstate__(self):
        if self.global_name:
            raise Exception("We expect that __reduce__ is used, not __getstate__.")
        raise Exception("Cannot pickle Entity object. (%r)" % self)


class OptionalNotImplementedError(NotImplementedError):
    """
    This can optionally be implemented, but it is not required by the API.
    """


def is_64bit_platform():
    """
    :return: True if we run on 64bit, False for 32bit
    :rtype: bool
    https://stackoverflow.com/questions/1405913/how-do-i-determine-if-my-python-shell-is-executing-in-32bit-or-64bit-mode-on-os
    """
    return sys.maxsize > 2**32


class BackendEngine:
    """
    Stores which backend engine we use in RETURNN.
    E.g. TensorFlow or PyTorch.
    """

    TensorFlowNetDict = 1
    TensorFlow = 2
    Torch = 3  # PyTorch
    selected_engine = None  # type: typing.Optional[int]  # One of the possible engines.

    @classmethod
    def select_engine(cls, *, engine=None, default_fallback_engine=None, config=None, _select_rf_backend: bool = True):
        """
        :param int engine: see the global class attribs for possible values
        :param int|None default_fallback_engine: if engine is None and not defined in config, use this
        :param returnn.config.Config config:
        :param _select_rf_backend: internal. avoids that Torch/TF/anything further gets imported at this point
        """
        if engine is None:
            if config is None:
                from returnn.config import get_global_config

                config = get_global_config()
            if config.value("backend", None):
                backend = config.value("backend", None)
                engine = {
                    "tensorflow-net-dict": cls.TensorFlowNetDict,
                    "tensorflow": cls.TensorFlow,
                    "torch": cls.Torch,
                }[backend]
            elif config.bool("use_tensorflow", False):
                engine = cls.TensorFlowNetDict
            if engine is None:
                if default_fallback_engine is not None:
                    engine = default_fallback_engine
                else:
                    raise Exception("No backend defined. Please set the config option 'backend'.")

        if _select_rf_backend:
            # noinspection PyProtectedMember
            from returnn.frontend import _backend

            if engine == cls.TensorFlow:
                _backend.select_backend_tf()
            elif engine == cls.TensorFlowNetDict:
                # Note that we assume that the user wants the RETURNN layers frontend (TF-based)
                # and not the low-level TF frontend.
                # If we want to expose the low-level TF frontend to the user directly at some point,
                # we would need a new config option.
                _backend.select_backend_returnn_layers_tf()
            elif engine == cls.Torch:
                _backend.select_backend_torch()
        cls.selected_engine = engine

    @classmethod
    def get_selected_engine(cls) -> int:
        """
        :return: one of the constants TensorFlowNetDict, TensorFlow, Torch
        """
        if cls.selected_engine is None:
            print("WARNING: BackendEngine.get_selected_engine() called before select_engine().", file=log.v3)
            cls.select_engine()
        return cls.selected_engine

    @classmethod
    def is_tensorflow_selected(cls):
        """
        :rtype: bool
        """
        return cls.get_selected_engine() in {cls.TensorFlowNetDict, cls.TensorFlow}

    @classmethod
    def is_torch_selected(cls):
        """
        :rtype: bool
        """
        return cls.get_selected_engine() == cls.Torch


class BehaviorVersion:
    """
    Stores the global behavior_version.

    The version will be set after the config is defined at __main__.init_config() or Engine.__init__().

    See :ref:`behavior_version`.
    """

    _latest_behavior_version = 21
    _behavior_version = None  # type: typing.Optional[int]
    _min_behavior_version = 0  # type: int

    @classmethod
    def set(cls, version):
        """
        :param int|None version:
        """
        if version == cls._behavior_version:
            return
        if cls._behavior_version is not None:
            raise Exception(f"behavior_version already set to {cls._behavior_version}, cannot reset to {version}")
        assert version >= cls._min_behavior_version
        if version > cls._latest_behavior_version:
            from returnn import __long_version__

            log.print_warning(
                "behavior_version %i > latest known behavior_version %i. "
                "Your RETURNN version is too old (%s)." % (version, cls._latest_behavior_version, __long_version__)
            )
        cls._behavior_version = version
        cls._handle_new_min_version()

    @classmethod
    def get(cls):
        """
        :rtype: int
        """
        if cls._behavior_version is not None:
            return cls._behavior_version
        return cls._min_behavior_version

    @classmethod
    def get_if_set(cls):
        """
        :rtype: int|None
        """
        return cls._behavior_version

    @classmethod
    def set_min_behavior_version(cls, min_behavior_version: int):
        """
        There are some RETURNN features which trigger a higher min behavior version.
        The min behavior version is used when no behavior version is explicitly set.
        But it is also an error if a behavior version is set, but it is lower than the min behavior version.

        :param min_behavior_version:
        """
        if min_behavior_version < cls._min_behavior_version:
            return  # just ignore
        assert min_behavior_version <= cls._latest_behavior_version
        if cls._behavior_version is not None:
            if cls._behavior_version < min_behavior_version:
                raise Exception(
                    f"set_min_behavior_version({min_behavior_version}) not allowed"
                    f" with explicit behavior_version {cls._behavior_version}"
                )
        cls._min_behavior_version = min_behavior_version
        cls._handle_new_min_version()

    @classmethod
    def is_set(cls):
        """
        :rtype: bool
        """
        return cls._behavior_version is not None

    class RequirementNotSatisfied(Exception):
        """
        Behavior version requirement is not satisfied
        """

    @classmethod
    def require(cls, condition, message, version):
        """
        :param bool condition:
        :param str message:
        :param int version:
        """
        assert version <= cls._latest_behavior_version
        if not condition:
            if BehaviorVersion.get() >= version:
                raise BehaviorVersion.RequirementNotSatisfied(
                    "%s (required since behavior_version >= %i)" % (message, version)
                )
            else:
                log.print_deprecation_warning(message, behavior_version=version)

    @classmethod
    def _get_state(cls):
        return cls._behavior_version, cls._min_behavior_version

    reset_callbacks: List[Callable[[], None]] = []
    handle_new_min_version_callbacks: List[Callable[[], None]] = []

    @classmethod
    def _reset(cls, state: Optional[Tuple[int, int]] = None):
        """
        Reset behavior version. This is only intended for internal use (e.g. testing).

        :param state: via :func:`_get_state`, or None to reset to initial state
        """
        if state:
            cls._behavior_version, cls._min_behavior_version = state
        else:
            cls._behavior_version = None
            cls._min_behavior_version = 0

        # Reset things we did in _handle_new_min_version.
        for cb in cls.reset_callbacks:
            cb()

    @classmethod
    def _handle_new_min_version(cls):
        """
        Callback, called when we know about a new min or exact behavior version.
        The version can only increase, unless :func:`_reset` is called.
        """
        # e.g. enable simple Dim equality check here...
        for cb in cls.handle_new_min_version_callbacks:
            cb()


def get_model_filename_postfix():
    """
    :return: one possible postfix of a file which will be present when the model is saved
    :rtype: str
    """
    assert BackendEngine.selected_engine is not None
    if BackendEngine.is_tensorflow_selected():
        # There will be multiple files but a *.meta file will always be present.
        return ".meta"
    if BackendEngine.is_torch_selected():
        return ".pt"
    return ""


def get_checkpoint_filepattern(filepath):
    """
    Removes optional .index or .meta extension

    :param str filepath:
    :return: CheckpointLoader compatible filepattern
    :rtype: str
    """
    if filepath.endswith(".meta"):
        return filepath[: -len(".meta")]
    elif filepath.endswith(".index"):
        return filepath[: -len(".index")]
    elif filepath.endswith(".pt"):
        return filepath[: -len(".pt")]
    return filepath


def sys_cmd_out_lines(s):
    """
    :param str s: shell command
    :rtype: list[str]
    :return: all stdout split by newline. Does not cover stderr.
    Raises CalledProcessError on error.
    """
    p = subprocess.Popen(
        s,
        stdout=subprocess.PIPE,
        shell=True,
        close_fds=True,
        env=dict(os.environ, LANG="en_US.UTF-8", LC_ALL="en_US.UTF-8"),
    )
    stdout = p.communicate()[0]
    if PY3:
        stdout = stdout.decode("utf8")
    result = [line.strip() for line in stdout.split("\n")[:-1]]
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
    return sys_exec_out("git", "rev-parse", "--short=%i" % length if length else "--short", commit, cwd=git_dir).strip()


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
        .replace(" ", ".")
    )


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


def describe_torch_version() -> str:
    """
    :return: Torch version and path info
    """
    try:
        # noinspection PyPackageRequirements
        import torch
    except ImportError as exc:
        return "<PyTorch ImportError: %s>" % exc
    try:
        tdir = os.path.dirname(torch.__file__)
    except Exception as e:
        tdir = "<unknown(exception: %r)>" % e
    version = getattr(torch, "__version__", "<unknown version>")
    version += " (%s)" % getattr(torch.version, "git_version", "<unknown git version>")
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

    return tuple([int(re.sub("(-rc[0-9]|-dev[0-9]*)", "", s)) for s in tf.__version__.split(".")])


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

    Also see :func:`expand_env_vars` or `os.path.expandvars` or :func:`string.Template.substitute` or
    :mod:`shlex` utils.
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


def expand_env_vars(s: str) -> str:
    """
    Similar as :func:`os.path.expandvars`:

    It replaces ``$var`` or ``${var}`` with the value of the environment variable ``var``.
    Also, ``$$`` is replaced by ``$``.
    Any usage of an undefined env vars will be an error.

    In addition to :func:`os.path.expandvars`,
    it handles ``$TMPDIR`` and ``$USER`` specially
    when they are not defined in ``os.environ``,
    by using :func:`get_temp_dir` or :func:`get_login_username`.

    Also see :func:`string.Template.substitute`.

    :param s: string with env vars like "$TMPDIR/$USER"
    :return: s with expanded env vars
    """

    # Code adapted from string.Template.substitute.
    delim = "$"
    delim_ = re.escape(delim)
    id_ = r"(?a:[_a-z][_a-z0-9]*)"
    pattern = rf"""
        {delim_}(?:
          (?P<escaped>{delim_})  |   # Escape sequence of two delimiters
          (?P<named>{id_})       |   # delimiter and a Python identifier
          (?P<invalid>)              # Other ill-formed delimiter exprs
        )
        """

    pattern_ = re.compile(pattern, re.VERBOSE | re.IGNORECASE)

    # Helper function for .sub()
    def _convert(mo: re.Match) -> str:
        # Check the most common path first.
        name = mo.group("named")
        if name is not None:
            if name in os.environ:
                return os.environ[name]
            if name in {"TMPDIR", "TEMP", "TMP"}:
                return get_temp_dir(with_username=False)
            if name == "USER":
                return get_login_username()
            raise ValueError(f"Undefined environment variable {name!r}")
        if mo.group("escaped") is not None:
            return delim
        if mo.group("invalid") is not None:
            i = mo.start("invalid")
            raise ValueError(f"Invalid placeholder in string: {s[i:i+2]!r}...")
        raise ValueError(f"Unrecognized named group in pattern {pattern}")

    return pattern_.sub(_convert, s)


def hdf5_dimension(filename, dimension):
    """
    :param str filename:
    :param str dimension:
    :rtype: numpy.ndarray|int
    """
    fin = h5py.File(filename, "r")
    if "/" in dimension:
        res = fin["/".join(dimension.split("/")[:-1])].attrs[dimension.split("/")[-1]]
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
    :rtype: int|None
    """
    # We could check via:
    # tf.contrib.framework.python.framework.checkpoint_utils.load_variable()
    # once we save that in the model.
    # See TFNetwork.Network._create_saver().
    # We don't have it in the model, though.
    # For now, just parse it from filename.
    # If TF, and symlink, resolve until no symlink anymore (e.g. if we symlinked the best epoch).
    for potential_ext in [".meta", ".pt"]:
        if not os.path.exists(filename + potential_ext):
            continue
        while os.path.exists(filename + potential_ext) and os.path.islink(filename + potential_ext):
            fn_with_ext_ = os.readlink(filename + potential_ext)
            assert fn_with_ext_.endswith(potential_ext), "strange? %s, %s" % (filename, potential_ext)
            filename = fn_with_ext_[: -len(potential_ext)]
        break
    m = re.match(".*\\.([0-9]+)", filename)
    if not m:
        return None
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
    except (io.UnsupportedOperation, ValueError):
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

            cr_ = struct.unpack("hh", fcntl.ioctl(fd, termios.TIOCGWINSZ, "1234"))  # noqa
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
        cr = (env.get("LINES", 25), env.get("COLUMNS", 80))
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
    return ("%." + str(prec) + "f") % (float(n) / (factor**i)) + postfixes[i]


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
            s,
            str(obj.dtype),
            obj.shape,
        )
    else:
        s = repr(obj)
    if limit is None:
        limit = _pretty_print_limit
    if len(s) > limit:
        s = s[: int(limit) - 3]
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
    if terminal_width <= 0:
        return
    if complete == 1.0:
        file.write("\r%s" % (terminal_width * " "))
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
    bars = "|" * int(complete * ntotal)
    spaces = " " * (ntotal - int(complete * ntotal))
    bar = bars + spaces
    file.write(
        "\r%s" % prefix + "[" + bar[: len(bar) // 2] + " " + progress + " " + bar[len(bar) // 2 :] + "]" + suffix
    )
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
        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        mem_bytes = 1024**4  # just some random number, 1TB
    return mem_bytes


def default_cache_size_in_gbytes(factor=0.7):
    """
    :param float|int factor:
    :rtype: int
    """
    mem_gbytes = available_physical_memory_in_bytes() / (1024.0**3)
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
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return "float('%s')" % repr(o)
    # fallback
    return repr(o)


def simple_obj_repr(obj):
    """
    :return: All self.__init__ args.
    :rtype: str
    """
    return obj.__class__.__name__ + "(%s)" % ", ".join(
        ["%s=%s" % (arg, better_repr(getattr(obj, arg))) for arg in getargspec(obj.__init__).args[1:]]
    )


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

    def __init__(self, dikt: Dict[str, Any]):
        """
        :param dikt:
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


def obj_diff_str(self, other, **kwargs):
    """
    :param object self:
    :param object other:
    :return: the difference described
    :rtype: str
    """
    diff_list = obj_diff_list(self, other, **kwargs)
    if not diff_list:
        return "No diff."
    return "\n".join(diff_list)


def obj_diff_list(self, other, **kwargs):
    """
    Note that we recurse to a certain degree to the items, but not fully.
    Some differences might just be summarized.

    :param object self:
    :param object other:
    :return: the difference described
    :rtype: list[str]
    """
    # Having them explicitly in kwargs because we cannot use `*` in Python 2,
    # but we do not want to allow them as positional args.
    prefix = kwargs.pop("_prefix", "")
    # If allowed_mapping(self, other),
    # they are assigned to the equal_mapping, if possible to do so unambiguously.
    allowed_mapping = kwargs.pop("allowed_mapping", None)
    equal_map_s2o = kwargs.pop("_equal_map_s2o", None)  # self -> other
    equal_map_o2s = kwargs.pop("_equal_map_o2s", None)  # other -> self
    equal_map_finished = kwargs.pop("_equal_map_finished", False)
    if kwargs:
        raise TypeError("obj_diff_list: invalid kwargs %r" % kwargs)

    if self is None and other is None:
        return []
    if self is None and other is not None:
        return ["%sself is None and other is %r" % (prefix, other)]
    if self is not None and other is None:
        return ["%sother is None and self is %r" % (prefix, self)]
    if type(self) != type(other):  # noqa
        return ["%stype diff: self is %s but other is %s" % (prefix, type(self).__name__, type(other).__name__)]

    if allowed_mapping:
        if equal_map_s2o is None:
            # Build up the unequal_map, by going through the objects.
            equal_map_s2o, equal_map_o2s = {}, {}
            obj_diff_list(
                self, other, allowed_mapping=allowed_mapping, _equal_map_s2o=equal_map_s2o, _equal_map_o2s=equal_map_o2s
            )
            equal_map_finished = True
    else:
        equal_map_finished = True
    if equal_map_s2o is None or equal_map_o2s is None:
        equal_map_s2o, equal_map_o2s = {}, {}  # simplifies the code below

    sub_kwargs = dict(
        allowed_mapping=allowed_mapping,
        _equal_map_s2o=equal_map_s2o,
        _equal_map_o2s=equal_map_o2s,
        _equal_map_finished=equal_map_finished,
    )

    if isinstance(self, (list, tuple)):
        assert isinstance(other, (list, tuple))
        if len(self) != len(other):
            return ["%slist diff len: len self: %i, len other: %i" % (prefix, len(self), len(other))]
        s = []
        for i, (a, b) in enumerate(zip(self, other)):
            s += obj_diff_list(a, b, _prefix="%s[%i] " % (prefix, i), **sub_kwargs)
        if s:
            return ["%slist diff:" % prefix] + s
        return []

    def _set_diff(a_, b_):
        # assume the values can be sorted
        a_ = sorted(a_)
        b_ = sorted(b_)
        self_diff_ = []
        same_ = []
        for v in a_:
            v_ = equal_map_s2o.get(v, v)
            if v_ in b_:
                same_.append(v)
                b_.remove(v_)
            else:
                self_diff_.append(v)
        other_diff_ = b_
        return self_diff_, same_, other_diff_

    if isinstance(self, set):
        assert isinstance(other, set)
        self_diff, _, other_diff = _set_diff(self, other)
        if len(self_diff) == len(other_diff) == 1:
            # potentially update equal_map
            s = obj_diff_list(list(self_diff)[0], list(other_diff)[0], **sub_kwargs)
            # ignore the potential updated equal_map now for simplicity. we will anyway do a second pass later.
            if len(s) == 1:
                return ["%sset diff value: %s" % (prefix, s[0])]
        s = []
        for key in self_diff:
            s += ["%s  %r not in other" % (prefix, key)]
        for key in other_diff:
            s += ["%s  %r not in self" % (prefix, key)]
        if s:
            return ["%sset diff:" % prefix] + s
        return []

    if isinstance(self, dict):
        assert isinstance(other, dict)
        self_diff, same, other_diff = _set_diff(self.keys(), other.keys())
        if not equal_map_finished and len(self_diff) == len(other_diff) == 1:
            # potentially update equal_map
            obj_diff_list(list(self_diff)[0], list(other_diff)[0], **sub_kwargs)
            # ignore the potential updated equal_map now for simplicity. we will anyway do a second pass later.
        s = []
        for key in self_diff:
            s += ["%s  key %r not in other" % (prefix, key)]
        for key in other_diff:
            s += ["%s  key %r not in self" % (prefix, key)]
        for key in same:
            key_ = equal_map_s2o.get(key, key)
            value_self = self[key]
            value_other = other[key_]
            s += obj_diff_list(value_self, value_other, _prefix="%s[%r] " % (prefix, key), **sub_kwargs)
        if s:
            return ["%sdict diff:" % prefix] + s
        return []

    if isinstance(self, np.ndarray):
        assert isinstance(other, np.ndarray)
        if not np.array_equal(self, other):
            return ["%sself: %r != other: %r" % (prefix, self, other)]
        return []

    if allowed_mapping and self != other and allowed_mapping(self, other):
        if self in equal_map_s2o:
            self = equal_map_s2o[self]
        elif other not in equal_map_o2s:  # don't map multiple times to this
            equal_map_s2o[self] = other
            equal_map_o2s[other] = self

    if self != other:
        return ["%sself: %r != other: %r" % (prefix, self, other)]
    return []


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
    if PY3:
        # These monkey patches are not necessary anymore. Nothing blocks signals anymore in Python 3.
        # https://github.com/albertz/playground/blob/master/thread-join-block.py
        # https://github.com/albertz/playground/blob/master/cond-wait-block.py
        return
    main_thread = threading.current_thread()
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
    is_main_thread = isinstance(threading.current_thread(), threading._MainThread)
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


def try_run(func, args=(), *, kwargs=None, catch_exc=Exception, default=None):
    """
    :param (()->T)|((X)->T) func:
    :param tuple args:
    :param dict|None kwargs:
    :param type[Exception] catch_exc:
    :param T2 default:
    :return: either ``func()`` or ``default`` if there was some exception
    :rtype: T|T2
    """
    # noinspection PyBroadException
    try:
        return func(*args, **(kwargs or {}))
    except catch_exc:
        return default


def validate_broadcast_all_sources(allow_broadcast_all_sources, inputs, common):
    """
    Call this when all inputs to some operation (layer) must be broadcasted.
    It checks whether broadcasting to all sources should be allowed.
    E.g. for input [B,T1,D1] + [B,T2,D2], when allowed, it would broadcast to [B,T1,T2,D1,D2].
    When not allowed, there must be at least one source where no broadcasting will be done.
    Whether it is allowed, this depends on the behavior version.
      https://github.com/rwth-i6/returnn/issues/691

    Common usages are for :func:`get_common_shape` or :func:`Data.get_common_data`.

    :param bool|NotSpecified allow_broadcast_all_sources:
    :param inputs: anything convertible to iterable of str, used for reporting
    :param common: anything convertible to str, used for reporting
    """
    msg = (
        "All inputs\n%s\nrequire broadcasting to \n  %s.\n" % ("\n".join(" - %s" % inp for inp in inputs), common)
        + "This must be explicitly allowed, e.g. by specifying out_shape."
    )
    if allow_broadcast_all_sources is NotSpecified:
        from returnn.util import BehaviorVersion

        BehaviorVersion.require(version=4, condition=False, message=msg)
        return
    if allow_broadcast_all_sources:
        return
    raise Exception(msg)


def class_idx_seq_to_1_of_k(seq, num_classes):
    """
    Basically one_hot.

    :param list[int]|np.ndarray seq:
    :param int num_classes:
    :rtype: np.ndarray
    """
    num_frames = len(seq)
    m = np.zeros((num_frames, num_classes), dtype="float32")
    m[np.arange(num_frames), seq] = 1
    return m


def uniq(seq):
    """
    Like Unix tool uniq. Removes repeated entries.
    See :func:`uniq_generic` for a generic (non-Numpy) version.

    :param numpy.ndarray seq:
    :return: seq
    :rtype: numpy.ndarray
    """
    diffs = np.ones_like(seq)
    diffs[1:] = seq[1:] - seq[:-1]
    idx = diffs.nonzero()
    return seq[idx]


def uniq_generic(seq):
    """
    Like Unix tool uniq. Removes repeated entries.
    See :func:`uniq` for an efficient Numpy implementation.
    See :func:`returnn.tf.util.basic.uniq` for an efficient TF implementation.

    :param list[T]|tuple[T] seq:
    :return: seq
    :rtype: list[T]
    """
    out = []
    visited = set()
    for x in seq:
        if x in visited:
            continue
        out.append(x)
        visited.add(x)
    return out


def slice_pad_zeros(x: numpy.ndarray, begin: int, end: int, axis: int = 0) -> numpy.ndarray:
    """
    :param x: of shape (..., time, ...)
    :param begin:
    :param end:
    :param axis:
    :return: basically x[begin:end] (with axis==0) but if begin < 0 or end > x.shape[0],
     it will not discard these frames but pad zeros, such that the resulting shape[0] == end - begin.
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


def random_orthogonal(shape, gain=1.0, seed=None):
    """
    Returns a random orthogonal matrix of the given shape.
    Code borrowed and adapted from Keras: https://github.com/fchollet/keras/blob/master/keras/initializers.py
    Reference: Saxe et al., https://arxiv.org/abs/1312.6120
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
    return gain * q[: shape[0], : shape[1]]


# noinspection PyUnusedLocal
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
    raise NotImplementedError("This feature was removed with dropped Theano support")


def prod(ls):
    """
    :param list[T]|tuple[T]|numpy.ndarray ls:
    :rtype: T|int|float
    """
    if len(ls) == 0:
        return 1
    x = ls[0]
    for y in ls[1:]:
        x = x * y  # *= doesn't work because x might be a tensor, and for e.g. torch.Tensor this op is in-place
    return x


def parse_orthography_into_symbols(
    orthography, upper_case_special=True, word_based=False, square_brackets_for_specials=True
):
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
                if ret and not ret[-1]:
                    ret[-1] += c
                else:
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


def parse_orthography(
    orthography, prefix=(), postfix=("[END]",), remove_chars="(){}", collapse_spaces=True, final_strip=True, **kwargs
):
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
    end_slashes_re = re.compile(r"(\\)*$")

    in_string = False
    in_multi = False
    in_single = False

    new_str = []
    index = 0

    for match in re.finditer(tokenizer, string):

        if not (in_multi or in_single):
            tmp = string[index : match.start()]
            if not in_string and strip_space:
                # replace white space as defined in standard
                tmp = re.sub("[ \t\n\r]+", "", tmp)
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
            if val == "/*":
                in_multi = True
            elif val == "//":
                in_single = True
        elif val == "*/" and in_multi and not (in_string or in_single):
            in_multi = False
        elif val in "\r\n" and not (in_multi or in_string) and in_single:
            in_single = False
        elif not ((in_multi or in_single) or (val in " \r\n\t" and strip_space)):
            new_str.append(val)

    new_str.append(string[index:])
    return "".join(new_str)


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
        return s.encode("utf-8")  # Python 2 str, Python 3 bytes
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
        self.max = self._max_error

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
            numbers_dict={k: const_number for k in numbers_dict.dict.keys()},
        )

    def copy_like(self, numbers_dict):
        """
        :param NumbersDict numbers_dict:
        :return: copy of self with same keys as numbers_dict as far as we have them
        :rtype: NumbersDict
        """
        if self.value is not None:
            return NumbersDict(
                broadcast_value=self.value if (numbers_dict.value is not None) else None,
                numbers_dict={k: self[k] for k in numbers_dict.dict.keys()},
            )
        else:
            return NumbersDict(
                broadcast_value=None, numbers_dict={k: self[k] for k in numbers_dict.dict.keys() if k in self.dict}
            )

    @property
    def keys_set(self):
        """
        Also see :func:`keys_union` if you want to have a deterministic order.

        :rtype: set[str]
        """
        return set(self.dict.keys())

    def keys_union(*number_dicts: NumbersDict) -> List[str]:
        """
        :return: union of keys over self and other. The order will be deterministic (unlike :func:`keys_set`)
        """
        seen = set()
        res = []
        for number_dict in number_dicts:
            for key in number_dict.dict.keys():
                if key not in seen:
                    seen.add(key)
                    res.append(key)
        return res

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
        # Keep consistent with self.__getitem__. If self.value is set, this will always be the default value.
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

    def has_value_for(self, key: str) -> bool:
        """
        :return: If self.value is set, always True, otherwise if key is in self.dict
        """
        return self.value is not None or key in self.dict

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
        for k in self.keys_union(other):
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
    def _max_error():
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
        return "%s(numbers_dict=%r, broadcast_value=%r)" % (self.__class__.__name__, self.dict, self.value)


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
                args = args[len(args) - len(arg_spec.defaults) :]
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


def type_attrib_mro_chain(cls: type, attr_name: str) -> list:
    """
    :return: list of all attributes with the given name in the MRO chain
    """
    attribs = []
    for cls_ in cls.mro():
        if attr_name in cls_.__dict__:
            attribs.append(cls_.__dict__[attr_name])
    return attribs


def next_type_attrib_in_mro_chain(cls: type, attr_name: str, attr):
    """
    :param cls:
    :param attr_name:
    :param attr: must be in the attrib MRO chain
    :return: next attribute in the MRO chain
    """
    attribs = type_attrib_mro_chain(cls, attr_name)
    assert attr in attribs
    idx = attribs.index(attr)
    assert idx + 1 < len(attribs)
    return attribs[idx + 1]


def custom_exec(source: str, source_filename: str, user_ns: Dict[str, Any], user_global_ns: Dict[str, Any]):
    """
    :param source:
    :param source_filename:
    :param user_ns:
    :param user_global_ns:
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
    if isinstance(obj, (list, tuple)):
        return tuple([make_hashable(item) for item in obj])
    if isinstance(obj, (str, unicode, float, int, long)):
        return obj
    if obj is None:
        return obj
    if "tensorflow" in sys.modules:
        import tensorflow as tf

        if isinstance(obj, tf.Tensor):
            return RefIdEq(obj)
    # Try if this is already hashable.
    try:
        hash(obj)
    except Exception:
        raise TypeError("don't know how to make hashable: %r (%r)" % (obj, type(obj)))
    return obj


class RefIdEq(Generic[T]):
    """
    Reference to some object (e.g. t.fTensor), but this object is always hashable,
    and uses the `id` of the function for the hash and equality.

    (In case of tf.Tensor, this is for compatibility
     because tf.Tensor.ref() was not available in earlier TF versions.
     However, we also need this for :class:`DictRefKeys`.)
    (This was TensorRef in earlier RETURNN versions.)
    """

    def __init__(self, obj: T):
        """
        :param obj: for example tf.Tensor
        """
        self.obj = obj

    def __repr__(self):
        return "TensorRef{%r}" % self.obj

    def __eq__(self, other):
        if other is None or not isinstance(other, RefIdEq):
            return False
        return self.obj is other.obj

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return id(self.obj)


class DictRefKeys(Generic[K, V]):
    """
    Like `dict`, but hash and equality of the keys
    """

    def __init__(self):
        self._d = {}  # type: Dict[RefIdEq[K], V]

    def __repr__(self):
        return "DictRefKeys(%s)" % ", ".join(["%r: %r" % (k, v) for (k, v) in self.items()])

    def items(self) -> Iterable[Tuple[K, V]]:
        """items"""
        for k, v in self._d.items():
            yield k.obj, v

    def keys(self) -> Iterable[K]:
        """keys"""
        for k in self._d.keys():
            yield k.obj

    def values(self) -> Iterable[V]:
        """values"""
        for v in self._d.values():
            yield v

    def __getitem__(self, item: K) -> V:
        return self._d[RefIdEq(item)]

    def __setitem__(self, key: K, value: V):
        self._d[RefIdEq(key)] = value

    def __contains__(self, item: K):
        return RefIdEq(item) in self._d


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
    return '"%s"' % s.replace("\\\\", "\\").replace("\n", "\\n").replace('"', '\\"').replace("'", "\\'")


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
    if isinstance(s, str) or "unicode" in str(type(s)):
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


def read_bytes_to_new_buffer(p: typing.BinaryIO, size: int) -> BytesIO:
    """
    Read bytes from stream s into a BytesIO buffer.
    Raises EOFError if not enough bytes are available.
    Then read it via :func:`read_pickled_object`.
    """
    stream = BytesIO()
    read_size = 0
    while read_size < size:
        data_raw = p.read(size - read_size)
        if len(data_raw) == 0:
            raise EOFError("expected to read %i bytes but got EOF after %i bytes" % (size, read_size))
        read_size += len(data_raw)
        stream.write(data_raw)
    stream.seek(0)
    return stream


def read_pickled_object(p: typing.BinaryIO) -> Any:
    """
    Read pickled object from stream p,
    after it was written via :func:`read_bytes_to_new_buffer`.

    :param p:
    """
    from returnn.util.task_system import Unpickler

    size_raw = read_bytes_to_new_buffer(p, 4).getvalue()
    (size,) = struct.unpack("<i", size_raw)
    assert size > 0, "read_pickled_object: We expect to get some non-empty package."
    stream = read_bytes_to_new_buffer(p, size)
    return Unpickler(stream).load()


def write_pickled_object(p: typing.BinaryIO, obj: Any):
    """
    Writes pickled object to stream p.
    """
    from returnn.util.task_system import Pickler

    stream = BytesIO()
    Pickler(stream).dump(obj)
    raw_data = stream.getvalue()
    assert len(raw_data) > 0
    p.write(struct.pack("<i", len(raw_data)))
    p.write(raw_data)
    p.flush()


def serialize_object(obj: Any) -> bytes:
    """
    Uses :func:`write_pickled_object`.
    """
    stream = BytesIO()
    write_pickled_object(stream, obj)
    return stream.getvalue()


def deserialize_object(data: bytes) -> Any:
    """
    Uses :func:`read_pickled_object`.
    """
    stream = BytesIO(data)
    return read_pickled_object(stream)


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

    def __init__(self, collection: Dict[str, Any], truth_value: Optional[bool] = None):
        """
        :param collection:
        :param truth_value: note: check explicitly for self.truth_value, bool(self) is not the same!
        """
        self.collection = collection
        if truth_value is None:
            truth_value = bool(self.collection)
        self.truth_value = truth_value
        self.got_items = set()

    def __repr__(self):
        return "%s(%r, truth_value=%r)" % (self.__class__.__name__, self.collection, self.truth_value)

    @classmethod
    def from_bool_or_dict(cls, value: Union[bool, Dict[str, Any]]) -> CollectionReadCheckCovered:
        """
        :param value:
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
        :rtype: T|typing.Any|None
        """
        try:
            return self[item]
        except KeyError:
            return default

    def __bool__(self) -> bool:  # Python 3
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


def which(program: str) -> Optional[str]:
    """
    Finds `program` in some of the dirs of the PATH env var.

    :param program: e.g. "python"
    :return: full path, e.g. "/usr/bin/python", or None
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
        postfix = basename[len("python") :]
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
    in_virtual_env = hasattr(sys, "real_prefix")  # https://stackoverflow.com/questions/1871549/
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
    :param str pkg_name: without version, e.g. just "tensorflow", or with version, e.g. "tensorflow==1.2.3"
    :rtype: bool
    """
    if not _pip_installed_packages:
        py = sys.executable
        pip_path = which_pip()
        cmd = [py, pip_path, "freeze"]
        for line in sys_exec_out(*cmd).splitlines():
            if line and "==" in line:
                if "==" not in pkg_name:
                    line = line[: line.index("==")]
                _pip_installed_packages.add(line)
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
        if args[: len(prefix_args)] == prefix_args:
            _original_execv(path, args)
        else:
            wrapped_execvpe(path, args)

    def execve(path, args, env):
        """
        :param str path:
        :param list[str]|tuple[str] args:
        :param dict[str] env:
        """
        if args[: len(prefix_args)] == prefix_args:
            _original_execve(path, args, env)
        else:
            wrapped_execvpe(path, args, env)

    def execl(file, *args):
        """execl(file, *args)

        Execute the executable file with argument list args, replacing the
        current process."""
        os.execv(file, args)

    def execle(file, *args):
        """execle(file, *args, env)

        Execute the executable file with argument list args and
        environment env, replacing the current process."""
        env = args[-1]
        os.execve(file, args[:-1], env)

    def execlp(file, *args):
        """execlp(file, *args)

        Execute the executable file (which is searched for along $PATH)
        with argument list args, replacing the current process."""
        os.execvp(file, args)

    def execlpe(file, *args):
        """execlpe(file, *args, env)

        Execute the executable file (which is searched for along $PATH)
        with argument list args and environment env, replacing the current
        process."""
        env = args[-1]
        os.execvpe(file, args[:-1], env)

    def execvp(file, args):
        """execvp(file, args)

        Execute the executable file (which is searched for along $PATH)
        with argument list args, replacing the current process.
        args may be a list or tuple of strings."""
        wrapped_execvpe(file, args)

    def execvpe(file, args, env):
        """execvpe(file, args, env)

        Execute the executable file (which is searched for along $PATH)
        with argument list args and environment env , replacing the
        current process.
        args may be a list or tuple of strings."""
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
        if v[0] == v[-1] == '"':
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

    if sys.platform == "win32":
        return os.getlogin()
    import pwd

    try:
        return pwd.getpwuid(os.getuid())[0]
    except KeyError:
        # pwd.getpwuid() can throw KeyError: 'getpwuid(): uid not found: 12345'
        # this can happen e.g. in a docker environment with mapped uids unknown to the docker OS
        return str(os.getuid())


def get_temp_dir(*, with_username: bool = True) -> str:
    """
    Similar as :func:`tempfile.gettempdir` but prefers `/var/tmp` over `/tmp`.

    :param with_username: whether to append the username to the path
    :return: e.g. "/var/tmp/$USERNAME"
    """
    postfix = ("/" + get_login_username()) if with_username else ""
    for envname in ["TMPDIR", "TEMP", "TMP"]:
        dirname = os.getenv(envname)
        if dirname:
            return dirname + postfix
    # /var/tmp should be more persistent than /tmp usually.
    if os.path.exists("/var/tmp"):
        return "/var/tmp" + postfix
    return "/tmp" + postfix


def get_cache_dir():
    """
    :return: used to cache non-critical things. by default get_temp_dir. unless you define env RETURNN_CACHE_DIR
    :rtype: str
    """
    if "RETURNN_CACHE_DIR" in os.environ:
        return os.environ["RETURNN_CACHE_DIR"]
    return get_temp_dir()


class LockFile:
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
            except OSError as exc:
                # Possible errors:
                # ENOENT (No such file or directory), e.g. if some parent directory was deleted.
                # EEXIST (File exists), if the dir already exists.
                if exc.errno not in [errno.ENOENT, errno.EEXIST]:
                    # Other error, so reraise.
                    # Common ones are e.g.:
                    # ENOSPC (No space left on device)
                    # EACCES (Permission denied)
                    raise
                # Ignore those errors.
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

    def __enter__(self) -> LockFile:
        self.lock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unlock()


def touch_file(filename: str, *, mode: int = 0o666):
    """
    If file does not exist, creates it,
    otherwise updates its mtime.

    :param filename:
    :param mode: if it does not exist, use given file permission mode
    """
    # Code adapted from pathlib.Path.touch.
    # First try to bump modification time
    # Implementation note: GNU touch uses the UTIME_NOW option of
    # the utimensat() / futimens() functions.
    try:
        os.utime(filename, None)
    except OSError:
        # Does not exist? Create now.
        flags = os.O_CREAT | os.O_WRONLY
        fd = os.open(filename, flags, mode)
        os.close(fd)

    # Check mtime.
    mtime = os.stat(filename).st_mtime
    assert 0 <= time.time() - mtime <= 10, f"mtime {mtime} of {filename} is too old, now {time.time()}"


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
            for sub_fn in glob(line[len("include ") :]):
                paths.extend(parse_ld_conf_file(sub_fn))
            continue
        paths.append(line)
    return paths


def get_ld_paths():
    """
    To be very correct, see man-page of ld.so.
    And here: https://unix.stackexchange.com/questions/354295/what-is-the-default-value-of-ld-library-path/354296
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
        line[len("hard resource_list:") :].strip()
        for line in stdout.splitlines()
        if line.startswith("hard resource_list:")
    ]
    assert len(ls) == 1
    opts = dict([opt.split("=", 1) for opt in ls[0].split(",")])  # noqa
    try:
        return int(opts["num_proc"])
    except ValueError as exc:
        raise Exception(
            "read_sge_num_procs: %r, invalid num_proc %r for job id %i.\nline: %r"
            % (exc, opts["num_proc"], job_id, ls[0])
        )


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
    key: value for (key, value) in [s.split("=", 1) for s in os.environ.get("THEANO_FLAGS", "").split(",") if s]
}


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
        for dev in config.list("device", []):
            if dev.startswith("gpu") or dev.startswith("cuda"):
                return True
            if dev == "all":
                return True
    return False


def get_cpu_model_name() -> str:
    """
    :return: e.g. "Intel(R) Core(TM) i5-8500 CPU @ 3.00GHz" via /proc/cpuinfo. falls back to platform.processor().
    """
    if os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo").read().splitlines():
            if not line.startswith("model name"):
                continue
            key, value = line.split(":", 1)
            key, value = key.strip(), value.strip()
            assert key == "model name"
            return value

    import platform

    return platform.processor()


def get_gpu_names():
    """
    :rtype: list[str]
    """
    if not _consider_check_for_gpu():
        return []
    if os.name == "nt":
        return "GeForce GTX 770"  # TODO
    elif sys.platform == "darwin":
        # TODO parse via xml output
        return sys_cmd_out_lines(
            "system_profiler SPDisplaysDataType | "
            "grep 'Chipset Model: NVIDIA' | "
            "sed 's/.*Chipset Model: NVIDIA *//;s/ *$//'"
        )
    else:
        try:
            return sys_cmd_out_lines("nvidia-smi -L | cut -d '(' -f 1 | cut -d ' ' -f 3- | sed -e 's/\\ $//'")
        except CalledProcessError:
            return []


def _get_num_gpu_devices():
    """
    :return: cpu,gpu
    :rtype: (int,int)
    """
    if os.name == "nt":
        return 1, 1  # TODO
    elif sys.platform == "darwin":
        return (
            int(sys_cmd_out_lines("sysctl -a | grep machdep.cpu.core_count | awk '{print $2}'")[0]),
            len(sys_cmd_out_lines("system_profiler SPDisplaysDataType | grep 'Chipset Model: NVIDIA' | cat")),
        )
    else:
        num_cpus = len(sys_cmd_out_lines("cat /proc/cpuinfo | grep processor")) or 1
        num_gpus = 0
        if _consider_check_for_gpu():
            try:
                num_gpus = len(sys_cmd_out_lines("nvidia-smi -L"))
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

        return get_func_str_from_code_object(frame.f_code, frame=frame)
    return fallback


def traceback_clear_frames(tb):
    """
    Clear traceback frame locals.

    Just like :func:`traceback.clear_frames`, but has an additional fix
    (https://github.com/python/cpython/issues/113939).

    :param types.TracebackType tb:
    """
    while tb:
        try:
            tb.tb_frame.clear()
        except RuntimeError:
            pass
        else:
            # Using this code triggers that the ref actually goes out of scope, otherwise it does not!
            # https://github.com/python/cpython/issues/113939
            tb.tb_frame.f_locals  # noqa
        tb = tb.tb_next


class InfiniteRecursionDetected(Exception):
    """
    Raised when an infinite recursion is detected, by guard_infinite_recursion.
    """


_guard_infinite_recursion_cache = threading.local()


@contextlib.contextmanager
def guard_infinite_recursion(*args):
    """
    Registers args (could be func + args) in some cache.
    If those args are already in the cache, it will raise an exception.

    It will use the id of the args as key and not use any hashing
    to allow that guard_infinite_recursion can be used
    to guard custom __hash__ implementations as well.
    """
    if not args:
        raise ValueError("guard_infinite_recursion needs at least one arg")
    key = tuple(id(arg) for arg in args)
    if not hasattr(_guard_infinite_recursion_cache, "cache"):
        _guard_infinite_recursion_cache.cache = set()
    if key in _guard_infinite_recursion_cache.cache:
        raise InfiniteRecursionDetected("infinite recursion detected")
    _guard_infinite_recursion_cache.cache.add(key)
    try:
        yield
    finally:
        _guard_infinite_recursion_cache.cache.remove(key)


def camel_case_to_snake_case(name):
    """
    :param str name: e.g. "CamelCase"
    :return: e.g. "camel_case"
    :rtype: str
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


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
    return get_hostname().startswith("cluster-cn-") or get_hostname().startswith("cn-")


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
    :param returnn.config.Config config:
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
                f.write("Returnn log file:\n" + "".join(["%s\n" % s for s in content]) + "\n")
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
                    "\n\n\n"
                    + "".join(
                        ["%s Config-file copied for logging purpose by Returnn.\n" % comment_prefix]
                        + ["%s %s\n" % (comment_prefix, s) for s in content]
                    )
                    + "\n"
                )
    except OSError as exc:
        if "Disk quota" in str(exc):
            print("log_runtime_info_to_dir: Error, cannot write: %s" % exc)
        else:
            raise


def should_write_to_disk(config):
    """
    :param returnn.config.Config config:
    :rtype: bool
    """
    if config.typed_value("torch_distributed") is not None:
        assert BackendEngine.is_torch_selected(), "torch_distributed assumes PyTorch"

        import torch.distributed

        if torch.distributed.get_rank() != 0:
            return False
    elif config.is_true("use_horovod"):
        assert BackendEngine.is_tensorflow_selected(), "use_horovod currently assumes TensorFlow"

        # noinspection PyPackageRequirements,PyUnresolvedReferences
        import horovod.tensorflow as hvd

        if hvd.rank() != 0:
            return False
    if config.is_true("dry_run"):
        return False
    return True


_default_global_inf_value = float("inf")


def get_global_inf_value() -> float:
    """
    :return: float("inf") by default, but tries to read `inf_value` from the global config
    """
    from returnn.config import get_global_config

    config = get_global_config(raise_exception=False)
    if not config:
        return _default_global_inf_value
    return config.float("inf_value", _default_global_inf_value)


def is_onnx_export_global() -> bool:
    """
    :return: False by default. If 'onnx_export' is set in the config, that value is used.
    """
    from returnn.config import get_global_config

    config = get_global_config(raise_exception=False)
    if not config:
        return False
    return config.bool("onnx_export", False)


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
    native = NativeCodeCompiler(base_name="patch_atfork", code_version=2, code=_c_code_patch_atfork, is_cpp=False)
    fn = native.get_lib_filename()
    return fn


def restart_returnn():
    """
    Restarts RETURNN.
    """
    log.flush()
    sys.stdout.flush()
    sys.stderr.flush()
    # https://stackoverflow.com/questions/72335904/simple-way-to-restart-application
    close_all_fds_except({0, 1, 2})
    os.execv(sys.executable, [sys.executable] + sys.argv)
    raise Exception("restart_returnn: execv failed")


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


def close_all_fds_except(except_fds):
    """
    Calls os.closerange except for the given fds.
    Code adopted and extended from multiprocessing.util.close_all_fds_except.

    :param typing.Collection[int] except_fds: usually at least {0,1,2}
    """
    # noinspection PyBroadException
    try:
        max_fd = os.sysconf("SC_OPEN_MAX")
    except Exception:
        max_fd = 256

    except_fds = sorted(list(except_fds) + [-1, max_fd])
    assert except_fds[0] == -1 and except_fds[-1] == max_fd, "fd invalid"

    for i in range(len(except_fds) - 1):
        if except_fds[i] + 1 < except_fds[i + 1]:
            os.closerange(except_fds[i] + 1, except_fds[i + 1])


def is_valid_fd(fd: int) -> bool:
    """
    :return: whether the file descriptor (fd) is still open and valid
    """
    # https://stackoverflow.com/questions/12340695/how-to-check-if-a-given-file-descriptor-stored-in-a-variable-is-still-valid
    import fcntl
    import errno

    try:
        fcntl.fcntl(fd, fcntl.F_GETFD)
        return True  # no matter what fcntl returned, it's a valid fd
    except OSError as exc:
        assert exc.errno == errno.EBADF
        return False


class Stats:
    """
    Collects mean and variance, running average.

    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """

    def __init__(self, *, format_str=None):
        """
        :param None|((float|numpy.ndarray)->str) format_str: used for __str__ and logging. :func:`str` by default.
            Could be e.g. :func:`human_bytes_size` for bytes.
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
                    self.total_data_len,
                    float(self.total_data_len) / self.num_seqs,
                )
            return "Stats(mean=%s, std_dev=%s, min=%s, max=%s, num_seqs=%i, %s)" % (
                self.format_str(self.get_mean()),
                self.format_str(self.get_std_dev()),
                self.format_str(self.min),
                self.format_str(self.max),
                self.num_seqs,
                extra_str,
            )
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
        m2 = m_a + m_b + mean_diff**2 * self.total_data_len * data.shape[0] / new_total_data_len
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
            print(
                "  %i seqs, %i total frames, %f average frames"
                % (self.num_seqs, self.total_data_len, self.total_data_len / float(self.num_seqs)),
                file=stream,
            )
        else:
            print("  %i seqs" % (self.num_seqs,), file=stream)
        if self.num_seqs > 0:
            print("  Mean: %s" % (self.format_str(self.get_mean()),), file=stream)
            print("  Std dev: %s" % (self.format_str(self.get_std_dev()),), file=stream)
            print("  Min/max: %s / %s" % (self.format_str(self.min), self.format_str(self.max)), file=stream)
            # print("Std dev (naive): %s" % numpy.sqrt(self.mean_sq - self.mean * self.mean), file=stream)
        else:
            print("  (No data)", file=stream)
        if output_file_prefix:
            assert self.num_seqs > 0, "cannot dump stats without any data"
            print("  Write mean/std-dev to %s.(mean|std_dev).txt." % (output_file_prefix,), file=stream)
            numpy.savetxt("%s.mean.txt" % output_file_prefix, self.get_mean())
            numpy.savetxt("%s.std_dev.txt" % output_file_prefix, self.get_std_dev())
            print("  Write min/max to %s.(min|max).txt." % (output_file_prefix,), file=stream)
            numpy.savetxt("%s.min.txt" % output_file_prefix, self.min)
            numpy.savetxt("%s.max.txt" % output_file_prefix, self.max)
            print("  Write extra info to %s.info.txt." % (output_file_prefix,), file=stream)
            with open("%s.info.txt" % output_file_prefix, "w") as f:
                print("{", file=f)
                print('"num_seqs": %i,' % self.num_seqs, file=f)
                print('"total_data_len": %i,' % self.total_data_len, file=f)
                print("}", file=f)


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


def ensure_list_of_type(ls, type_):
    """
    :param list ls:
    :param (()->T)|type[T] type_: type of instances of `ls`.
      Note the strange type here in the docstring is due to some PyCharm type inference problems
      (https://youtrack.jetbrains.com/issue/PY-50828).
    :rtype: list[T]
    """
    assert all(isinstance(elem, type_) for elem in ls)
    return ls


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
            ngram = tuple(segment[i : i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4, use_bp=True):
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

    for references, translations in zip(reference_corpus, translation_corpus):
        reference_length += len(references)
        translation_length += len(translations)
        ref_ngram_counts = _get_ngrams(references, max_order)
        translation_ngram_counts = _get_ngrams(translations, max_order)

        overlap = {ngram: min(count, translation_ngram_counts[ngram]) for ngram, count in ref_ngram_counts.items()}

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
            bp = math.exp(1 - 1.0 / ratio)
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
    prefix_dir = ""
    if not os.path.exists(filename):
        assert filename[0] != "/"
        # Maybe relative to Returnn?
        prefix_dir = "%s/" % returnn_root_dir
    assert os.path.exists(prefix_dir + filename)
    assert filename.endswith(".py") or os.path.isdir(prefix_dir + filename)
    dirs = filename.split("/")
    dirs, base_fn = dirs[:-1], dirs[-1]
    assert len(dirs) >= 1
    for i in reversed(range(len(dirs))):
        d = prefix_dir + "/".join(dirs[: i + 1])
        assert os.path.isdir(d)
        if os.path.exists("%s/__init__.py" % d):
            continue
        if d not in sys.path:
            sys.path.append(d)
        m = ".".join(dirs[i + 1 :] + [base_fn])
        if base_fn.endswith(".py"):
            m = m[:-3]
        return importlib.import_module(m)
    raise ValueError("cannot figure out base module path from %r" % filename)


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
    for line in open("/proc/%i/maps" % pid, "r").read().splitlines():  # for each mapped region
        # https://stackoverflow.com/questions/1401359/understanding-linux-proc-id-maps
        # address           perms offset  dev   inode   pathname
        # E.g.:
        # 7ff2de91c000-7ff2de91e000 rw-p 0017c000 08:02 794844                     /usr/lib/x86_64-linux-gnu/libstdc+...
        m = re.match(
            r"^([0-9A-Fa-f]+)-([0-9A-Fa-f]+)\s+([rwxps\-]+)\s+([0-9A-Fa-f]+)\s+([0-9A-Fa-f:]+)\s+([0-9]+)\s*(.*)$", line
        )
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


@contextlib.contextmanager
def override_env_var(var_name: str, value: str):
    """
    context manager for temporarily overriding the value of an env var

    :param var_name: the name of the environment variable to override
    :param value: the value to set while the context mgr is active
    """

    cur_val = os.environ.get(var_name)
    os.environ[var_name] = value
    try:
        yield
    finally:
        if cur_val is not None:
            os.environ[var_name] = cur_val
        else:
            os.environ.pop(var_name)


fwd_compatibility_rng = np.random.default_rng()


def get_fwd_compat_kwargs() -> Dict[str, Any]:
    """
    Get randomly named kwargs for ensuring forwards compatibility in user code.
    """
    i = fwd_compatibility_rng.integers(0, 100)
    return {f"__fwd_compat_random_arg_{i:03}": None}
