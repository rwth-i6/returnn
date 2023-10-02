"""
Python extension module compiler
"""

from __future__ import annotations

import sys
import sysconfig
from .native_code_compiler import NativeCodeCompiler


class PyExtModCompiler(NativeCodeCompiler):
    """
    Python extension module compiler
    """

    CacheDirName = "returnn_py_ext_mod_cache"

    def __init__(self, include_paths=(), **kwargs):
        py_compile_vars = sysconfig.get_config_vars()
        include_paths = list(include_paths) + [py_compile_vars["INCLUDEPY"]]
        super().__init__(
            include_paths=include_paths,
            **kwargs,
        )
        self._py_compile_vars = py_compile_vars
        self._py_mod = None

    _relevant_info_keys = NativeCodeCompiler._relevant_info_keys + ("py_version",)

    def _extra_common_opts(self):
        base_flags = super()._extra_common_opts()
        py_compile_flags = self._py_compile_vars["CFLAGS"].split() if self._py_compile_vars["CFLAGS"] else []
        return base_flags + py_compile_flags

    def _make_info_dict(self):
        d = super()._make_info_dict()
        d.update({"py_version": sys.version_info[:2]})
        return d

    def load_py_module(self):
        """
        :return: Python extension module
        """
        from importlib.util import spec_from_loader, module_from_spec
        from importlib.machinery import ExtensionFileLoader

        if self._py_mod:
            return self._py_mod

        self._maybe_compile()
        mod_name = self.base_name

        spec = spec_from_loader(mod_name, ExtensionFileLoader(mod_name, self._so_filename))  # noqa
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)  # noqa
        self._py_mod = mod
        return mod
