"""
Native code as Python extension module for the RETURNN frontend, including tensor methods and ops.
"""

from __future__ import annotations

import os
import hashlib
from glob import glob
from returnn.util.py_ext_mod_compiler import PyExtModCompiler

_module = None
_my_dir = os.path.dirname(os.path.abspath(__file__))


def get_module(*, verbose: bool = False):
    """
    :return: native Python extension module
    """
    global _module
    if _module and not verbose:
        return _module

    # Single source, which includes all cpp files, and also includes the hash of all hpp/cpp files.
    src_code = ""
    for fn in sorted(glob(_my_dir + "/*.hpp")):
        src_code += f"// {os.path.basename(fn)} code hash md5: {_code_hash_md5(fn)}\n"
    for fn in sorted(glob(_my_dir + "/*.cpp")):
        src_code += f"// {os.path.basename(fn)} code hash md5: {_code_hash_md5(fn)}\n"
        src_code += f'#include "{os.path.basename(fn)}"\n'

    compiler = PyExtModCompiler(
        base_name="_returnn_frontend_native",
        code_version=1,
        code=src_code,
        include_paths=(_my_dir,),
        is_cpp=True,
        verbose=verbose,
    )
    module = compiler.load_py_module()
    if not _module:
        _module = module
    return _module


def _code_hash_md5(filename: str) -> str:
    f_code = open(filename).read()
    h = hashlib.md5()
    h.update(f_code.encode("utf8"))
    return h.hexdigest()


_is_set_up = False


def setup():
    """
    Setup the native code.
    """
    global _is_set_up
    if _is_set_up:
        return
    _is_set_up = True  # only try once

    from returnn.tensor import Tensor, Dim

    # First we can use some existing native variants, which do not require our own native code.
    # The raw_tensor getter is replaced here, the raw_tensor setter is replaced below.
    Tensor.raw_tensor = property(Tensor._raw_tensor.__get__, Tensor.raw_tensor.__set__)  # noqa
    Dim.dimension = property(Dim.size.__get__)  # noqa

    try:
        mod = get_module()
    except Exception as exc:
        if os.environ.get("RETURNN_TEST") == "1":
            raise
        print("RETURNN frontend _native backend: Error while getting module:")
        print(exc)
        print("This is optional (although very recommended), so we continue without it.")
        return

    Tensor.raw_tensor = property(Tensor._raw_tensor.__get__, mod.tensor_raw_tensor_setter)  # noqa
    Tensor._raw_backend = property(mod.get_backend_for_tensor)  # noqa

    # noinspection PyProtectedMember
    from returnn.tensor.tensor import _TensorOpOverloadsMixin

    for name, cur_func in _TensorOpOverloadsMixin.__dict__.items():
        if not callable(cur_func):
            continue
        assert name.startswith("__") and name.endswith("__")
        native_func = getattr(mod, "_tensor_" + name[2:-2] + "_instancemethod")
        assert callable(native_func)
        setattr(_TensorOpOverloadsMixin, name, native_func)

    import returnn.frontend as rf
    from returnn.frontend import math_ as rf_math

    for rf_name, native_name in {
        "compare": "tensor_compare",
        "combine": "tensor_combine",
        "equal": "tensor_eq",
        "not_equal": "tensor_ne",
        "less": "tensor_lt",
        "less_equal": "tensor_le",
        "greater": "tensor_gt",
        "greater_equal": "tensor_ge",
        "add": "tensor_add",
        "sub": "tensor_sub",
        "mul": "tensor_mul",
        "true_divide": "tensor_truediv",
        "floor_divide": "tensor_floordiv",
        "neg": "tensor_neg",
        "mod": "tensor_mod",
        "pow": "tensor_pow",
        "logical_and": "tensor_and",
        "logical_or": "tensor_or",
        "logical_not": "tensor_invert",
        "abs": "tensor_abs",
        "ceil": "tensor_ceil",
        "floor": "tensor_floor",
    }.items():
        assert hasattr(rf, rf_name)
        assert hasattr(rf_math, rf_name)
        native_func = getattr(mod, native_name)
        setattr(rf, rf_name, native_func)
        setattr(rf_math, rf_name, native_func)
