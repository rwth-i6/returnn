"""
Native code compiler
"""

from __future__ import annotations
from typing import Optional, Union, Sequence, List, Tuple, Dict
import typing
import os
import sys
import shutil

from . import basic as util


class NativeCodeCompiler:
    """
    Helper class to compile native C/C++ code on-the-fly.
    """

    CacheDirName = "returnn_native"
    CollectedCompilers: Optional[List[NativeCodeCompiler]] = None

    def __init__(
        self,
        base_name: str,
        *,
        code_version: Union[int, Tuple[int, ...]] = 1,
        code: str,
        is_cpp: bool = True,
        c_macro_defines: Optional[Dict[str, Union[str, int, None]]] = None,
        ld_flags: Optional[Sequence[str]] = None,
        include_paths: Optional[Sequence[str]] = (),
        include_deps: Optional[Sequence[str]] = None,
        static_version_name: Optional[str] = None,
        should_cleanup_old_all: bool = True,
        should_cleanup_old_mydir: bool = False,
        use_cxx11_abi: bool = False,
        log_stream: Optional[typing.TextIO] = None,
        verbose: Optional[bool] = None,
    ):
        """
        :param base_name: base name for the module, e.g. "zero_out"
        :param code_version: check for the cache whether to reuse
        :param code: the source code itself
        :param is_cpp: if False, C is assumed
        :param c_macro_defines: e.g. {"TENSORFLOW": 1}
        :param ld_flags: e.g. ["-lblas"]
        :param include_paths:
        :param include_deps: if provided and an existing lib file,
            we will check if any dependency is newer
            and we need to recompile. we could also do it automatically via -MD but that seems overkill and too slow.
        :param static_version_name: normally, we use .../base_name/hash as the dir
            but this would use .../base_name/static_version_name.
        :param should_cleanup_old_all: whether we should look in the cache dir
            and check all ops if we can delete some old ones which are older than some limit
            (self._cleanup_time_limit_days)
        :param should_cleanup_old_mydir: whether we should delete our op dir before we compile there.
        :param log_stream: file stream for print statements
        :param verbose: be slightly more verbose
        """
        if self.CollectedCompilers is not None:
            self.CollectedCompilers.append(self)
        if verbose is None:
            verbose = os.environ.get("RETURNN_NATIVE_CODE_COMPILER_VERBOSE") == "1"
        self.verbose = verbose
        self.cache_dir = "%s/%s" % (util.get_cache_dir(), self.CacheDirName)
        self._include_paths = list(include_paths)
        self.base_name = base_name
        self.code_version = code_version
        self.code = code
        self.is_cpp = is_cpp
        self.c_macro_defines = {k: v for k, v in (c_macro_defines or {}).items() if v is not None}
        self.ld_flags = ld_flags or []
        self.include_deps = include_deps
        self.static_version_name = static_version_name
        self.use_cxx11_abi = use_cxx11_abi
        self._code_hash = self._make_code_hash()
        self._info_dict = self._make_info_dict()
        self._hash = self._make_hash()
        self._ctypes_lib = None
        if should_cleanup_old_all:
            self._cleanup_old()
        self._should_cleanup_old_mydir = should_cleanup_old_mydir
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
            lock = util.LockFile(full_dir_path)
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
                self._cleanup_old_path(full_dir_path, reason="%s old" % util.hms(dt))

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

    _relevant_info_keys = (
        "code_version",
        "code_hash",
        "c_macro_defines",
        "ld_flags",
        "compiler_bin",
        "platform",
        "use_cxx11_abi",
        "cpp_version",
    )

    def _make_info_dict(self):
        """
        :rtype: dict[str]
        """
        import platform

        return {
            "base_name": self.base_name,
            "include_paths": self._include_paths,
            "code_version": self.code_version,
            "code_hash": self._code_hash,
            "c_macro_defines": self.c_macro_defines,
            "ld_flags": self.ld_flags,
            "compiler_bin": self._get_compiler_bin(),
            "platform": platform.platform(),
            "use_cxx11_abi": self.use_cxx11_abi,
            "cpp_version": self.cpp_version,
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
            f.write("%s\n" % util.better_repr(self._info_dict))

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
        lock = util.LockFile(self._mod_path)
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
            return get_cpp_bin()
        return get_cc_bin()

    def _transform_compiler_opts(self, opts):
        """
        :param list[str] opts:
        :rtype: list[str]
        """
        return opts

    cpp_version = 11

    def _extra_common_opts(self):
        """
        :rtype: list[str]
        """
        if self.is_cpp:
            return [f"-std=c++{self.cpp_version}"]
        return []

    def _transform_ld_flags(self, opts: Sequence[str]) -> Sequence[str]:
        res = []
        for opt in opts:
            if opt.startswith("-l") or opt.startswith("-L"):
                res.append(opt)
            else:
                res.append("-Wl," + opt)
        opts = res
        if sys.platform == "darwin":
            res = []
            for opt in opts:
                # It seems some versions of MacOS ld cannot handle the `-l:filename` argument correctly.
                # E.g. TensorFlow 1.14 incorrectly uses this.
                # https://github.com/tensorflow/tensorflow/issues/30564
                if opt.startswith("-l:lib") and opt.endswith(".dylib"):
                    opt = "-l%s" % opt[len("-l:lib") : -len(".dylib")]
                res.append(opt)
            return res
        return opts

    def _maybe_compile_inner(self):
        # Directory should be created by the locking mechanism.
        assert os.path.exists(self._mod_path)
        with open(self._c_filename, "w") as f:
            f.write(self.code)
        common_opts = ["-shared", "-O2"]
        common_opts += self._extra_common_opts()
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
        opts += self._transform_ld_flags(self.ld_flags)
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


def get_cc_bin() -> str:
    """
    :return: path
    """
    cc_bin = os.environ.get("CC", "")
    if cc_bin:
        if cc_bin.startswith("/"):
            return cc_bin
        cc_bin = shutil.which(cc_bin)
        if cc_bin:
            return cc_bin
    cc_bin = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if not cc_bin:
        raise RuntimeError("Cannot find C compiler (cc, clang, gcc) in PATH")
    return cc_bin


def get_cpp_bin() -> str:
    """
    :return: path
    """
    cpp_bin = os.environ.get("CXX", "")
    if cpp_bin:
        if cpp_bin.startswith("/"):
            return cpp_bin
        cpp_bin = shutil.which(cpp_bin)
        if cpp_bin:
            return cpp_bin
    cpp_bin = shutil.which("c++") or shutil.which("cpp") or shutil.which("clang++") or shutil.which("g++")
    if not cpp_bin:
        raise RuntimeError("Cannot find C++ compiler (c++, cpp, clang++, g++) in PATH")
    return cpp_bin
