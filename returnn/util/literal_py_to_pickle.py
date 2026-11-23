"""
Convert literal Python (Python code which can be parsed with ``ast.literal_eval``
to Pickle.

Code partly taken from here:
https://github.com/albertz/literal-python-to-pickle
"""

from __future__ import annotations
from typing import Optional
import pickle
import ctypes
import os
from .basic import NativeCodeCompiler


def literal_eval(s):
    """
    This can be used as an alternative to ``ast.literal_eval``.
    In contrast to ``ast.literal_eval``, it also accepts bytes,
    and it should be ~5x faster.

    :param str|bytes s:
    :return: any object
    """
    raw_pickle = py_to_pickle(s)
    return pickle.loads(raw_pickle)


_lib: Optional[ctypes.CDLL] = None


def py_to_pickle(s):
    """
    :param str|bytes s:
    :rtype: bytes
    """
    if isinstance(s, bytes):
        in_bytes = s
    else:
        assert isinstance(s, str)
        in_bytes = s.encode("utf8")
    in_len = len(in_bytes)
    out_len = in_len + 1000  # usually enough, else we retry with the needed size below
    out_ = ctypes.create_string_buffer(out_len)
    out_size = ctypes.c_size_t(0)

    global _lib
    if not _lib:
        _lib = ctypes.CDLL(_get_native_lib_filename())
        _lib.py_to_pickle.argtypes = (
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        )
        _lib.py_to_pickle.restype = ctypes.c_int

    res = _lib.py_to_pickle(in_bytes, in_len, out_, out_len, ctypes.byref(out_size))
    if res == 2:  # out_ too small
        out_len = out_size.value
        out_ = ctypes.create_string_buffer(out_len)
        res = _lib.py_to_pickle(in_bytes, in_len, out_, out_len, ctypes.byref(out_size))
    assert res == 0, "there was some error"
    return ctypes.string_at(out_, out_size.value)


_my_dir = os.path.dirname(os.path.abspath(__file__))
_native_cpp_filename = _my_dir + "/py-to-pickle.cpp"
_native_lib_filename: Optional[str] = None


def _get_native_lib_filename():
    """
    :return: path to our patch_atfork lib. see :func:`maybe_restart_returnn_with_atfork_patch`
    :rtype: str
    """
    global _native_lib_filename
    if _native_lib_filename:
        return _native_lib_filename
    native = NativeCodeCompiler(
        base_name="pytopickle",
        code_version=2,
        code=open(_native_cpp_filename).read(),
        is_cpp=True,
        c_macro_defines={"LIB": 1},
    )
    _native_lib_filename = native.get_lib_filename()
    return _native_lib_filename
