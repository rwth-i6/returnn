"""
Native code as Python extension module for the RETURNN frontend, including tensor methods and ops.
"""

from __future__ import annotations

import os
from glob import glob
from returnn.util.py_ext_mod_compiler import PyExtModCompiler


_module = None
_my_dir = os.path.dirname(os.path.abspath(__file__))


def get_module():
    """
    :return: native Python extension module
    """
    global _module
    if _module:
        return _module

    # Put code all together in one big blob.
    # (Similar logic as in ken_lm.get_tf_mod.)
    files = sorted(glob(_my_dir + "/*.cpp"))
    src_code = ""
    for fn in files:
        f_code = open(fn).read()
        src_code += "\n// ------------ %s : BEGIN { ------------\n" % os.path.basename(fn)
        # https://gcc.gnu.org/onlinedocs/cpp/Line-Control.html#Line-Control
        src_code += '#line 1 "%s"\n' % os.path.basename(fn)
        src_code += f_code
        src_code += "\n// ------------ %s : END } --------------\n\n" % os.path.basename(fn)

    compiler = PyExtModCompiler(
        base_name="_returnn_frontend_native",
        code_version=1,
        code=src_code,
        include_paths=(_my_dir,),
        is_cpp=True,
    )
    _module = compiler.load_py_module()
    return _module


def setup():
    """
    Setup the native code.
    """
    try:
        mod = get_module()
    except Exception as exc:
        if os.environ.get("RETURNN_TEST") == "1":
            raise
        print("RETURNN frontend _native backend: Error while getting module:")
        print(exc)
        print("This is optional (although very recommended), so we continue without it.")
        return

    from returnn.tensor import Tensor

    Tensor._raw_backend = property(mod.get_backend_for_tensor)  # noqa
