"""
Convert literal Python (Python code which can be parsed with ``ast.literal_eval``
to Pickle.

Code partly taken from here:
https://github.com/albertz/literal-python-to-pickle
"""

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
    in_ = ctypes.create_string_buffer(in_bytes)
    in_len = len(in_bytes)
    out_len = in_len + 1000  # should always be enough (some buffer len + len of literal Python code)
    out_ = ctypes.create_string_buffer(out_len)

    lib = ctypes.CDLL(_get_native_lib_filename())
    lib.py_to_pickle.argtypes = (ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t)
    lib.py_to_pickle.restype = ctypes.c_int

    res = lib.py_to_pickle(in_, in_len, out_, out_len)
    assert res == 0, "there was some error"
    return out_.raw


_my_dir = os.path.dirname(os.path.abspath(__file__))
_native_cpp_filename = _my_dir + "/py-to-pickle.cpp"
_native_lib_filename = None


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
        code_version=1,
        code=open(_native_cpp_filename).read(),
        is_cpp=True,
        c_macro_defines={"LIB": 1},
    )
    _native_lib_filename = native.get_lib_filename()
    return _native_lib_filename
