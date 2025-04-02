"""
Native code as Python extension module for the RETURNN frontend, including tensor methods and ops.
"""

from __future__ import annotations

import os
import hashlib
from glob import glob
import textwrap
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

    if os.environ.get("RETURNN_TEST") == "1":
        src_code = (
            textwrap.dedent(
                """\
                #define DEBUG 1
                #ifdef NDEBUG
                #undef NDEBUG
                #endif

                """
            )
            + src_code
        )
        verbose = True

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

    # noinspection PyProtectedMember
    from returnn.tensor.tensor import _TensorOpOverloadsMixin, _TensorMixin

    # noinspection PyProtectedMember
    from returnn.tensor.dim import _DimMixin

    # First we can use some existing native variants, which do not require our own native code.
    # The raw_tensor getter is replaced here, the raw_tensor setter is replaced below.
    Tensor.raw_tensor = property(Tensor._raw_tensor.__get__, Tensor.raw_tensor.__set__)  # noqa
    _TensorMixin.placeholder = Tensor.raw_tensor
    Tensor.dims = property(Tensor._dims.__get__)  # noqa
    _TensorMixin.dim_tags = Tensor.dims  # noqa
    _DimMixin.dimension = property(Dim.size.__get__)  # noqa

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
    _TensorMixin.placeholder = Tensor.raw_tensor
    _TensorMixin._raw_backend = property(mod.get_backend_for_tensor)  # noqa

    # Functions in _TensorOpOverloadsMixin are replaced with native variants.
    for name, cur_func in _TensorOpOverloadsMixin.__dict__.items():  # just all of them
        if not callable(cur_func):
            continue
        if name in {"__bool__"}:  # some exceptions
            continue
        assert name.startswith("__") and name.endswith("__")
        native_func = getattr(mod, "_tensor_" + name[2:-2] + "_instancemethod")
        assert callable(native_func)
        setattr(_TensorOpOverloadsMixin, name, native_func)

    # Functions in _TensorMixin are replaced with native variants.
    for rf_name, native_name in {
        "copy": "tensor_copy",
        "copy_template": "tensor_copy_template",
        "get_out_permutation_to_dims": "tensor_get_out_permutation_to_dims",
        "copy_compatible_to_dims": "tensor_copy_compatible_to_dims",
        "copy_compatible_to_dims_raw": "tensor_copy_compatible_to_dims_raw",
    }.items():
        assert hasattr(_TensorMixin, rf_name)
        native_func = getattr(mod, "_" + native_name + "_instancemethod")
        setattr(_TensorMixin, rf_name, native_func)

    import returnn.frontend as rf
    from returnn.frontend import math_ as rf_math

    # Functions in rf and rf.math_ are replaced with native variants.
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


_is_set_up_torch = False


def setup_torch():
    """
    Like :func:`setup`, but specifically for the PyTorch backend.
    This assumes that we can `import torch`, unlike :func:`setup`.
    """
    global _is_set_up_torch
    if _is_set_up_torch:
        return
    _is_set_up_torch = True  # only try once

    import torch

    # noinspection PyBroadException
    try:
        mod = get_module()
    except Exception:
        if os.environ.get("RETURNN_TEST") == "1":
            raise
        # No message here, we expect that setup() was already called and printed the message.
        return

    from returnn.torch.frontend import TorchBackend

    TorchBackend.executing_eagerly = True.__bool__
    TorchBackend.get_dtype_name_raw = mod.raw_torch_tensor_get_dtype
    TorchBackend.get_ndim_raw = staticmethod(torch.Tensor.dim)
    TorchBackend.expand_dims_raw = staticmethod(torch.unsqueeze)
    TorchBackend.reshape_raw = staticmethod(torch.reshape)
