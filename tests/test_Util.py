# -*- coding: utf8 -*-

import _setup_test_env  # noqa
from numpy.testing import assert_almost_equal
from returnn.util.basic import *
import sys
import os
import numpy as np
import numpy
import pytest
import unittest
import textwrap
import signal

from returnn.util import better_exchook
from returnn.util.py_ext_mod_compiler import PyExtModCompiler

better_exchook.replace_traceback_format_tb()

my_dir = os.path.dirname(os.path.realpath(__file__))


def _sig_alarm_handler(signum, frame):
    raise Exception(f"Alarm (timeout) signal handler")


signal.signal(signal.SIGALRM, _sig_alarm_handler)


@contextlib.contextmanager
def timeout(seconds=10):
    """
    :param seconds: when the context is not closed within this time, an exception will be raised
    """
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def _get_tmp_dir() -> str:
    """
    :return: dirname
    """
    import tempfile
    import shutil
    import atexit

    name = tempfile.mkdtemp()
    assert name and os.path.isdir(name) and not os.listdir(name)
    atexit.register(lambda: shutil.rmtree(name))
    return name


def test_cmd_true():
    r = sys_cmd_out_lines("true")
    assert r == []


def test_cmd_false():
    with pytest.raises(CalledProcessError):
        sys_cmd_out_lines("false")


def test_cmd_stdout():
    r = sys_cmd_out_lines("echo 1; echo 2;")
    assert r == ["1", "2"]


def test_cmd_stderr():
    r = sys_cmd_out_lines("echo x >&2")
    assert r == [], "cmd() output should only cover stdout"


def test_hms():
    assert hms(5) == "0:00:05"
    assert hms(65) == "0:01:05"
    assert hms(65 + 60 * 60) == "1:01:05"


def test_hms_fraction():
    assert hms_fraction(0, decimals=3) == "0:00:00.000"
    assert hms_fraction(5, decimals=3) == "0:00:05.000"
    assert hms_fraction(5.345, decimals=3) == "0:00:05.345"
    assert hms_fraction(65.345, decimals=3) == "0:01:05.345"


def test_uniq():
    assert (uniq(np.array([0, 1, 1, 1, 2, 2])) == np.array([0, 1, 2])).all()


def test_slice_pad_zeros():
    assert list(slice_pad_zeros(np.array([1, 2, 3, 4]), begin=1, end=3)) == [2, 3]
    assert list(slice_pad_zeros(np.array([1, 2, 3, 4]), begin=-2, end=2)) == [0, 0, 1, 2]
    assert list(slice_pad_zeros(np.array([1, 2, 3, 4]), begin=-2, end=6)) == [0, 0, 1, 2, 3, 4, 0, 0]
    assert list(slice_pad_zeros(np.array([1, 2, 3, 4]), begin=2, end=6)) == [3, 4, 0, 0]


def test_math_PiecewiseLinear():
    from returnn.util.math import PiecewiseLinear

    eps = 1e-5
    f = PiecewiseLinear({1: 2, 3: 4, 5: 1})
    assert str(f) == "PiecewiseLinear({1: 2, 3: 4, 5: 1})"
    assert f(0) == 2
    assert f(1 - eps) == 2
    assert f(1) == 2
    assert_almost_equal(f(1 + eps), 2, decimal=4)
    assert f(2) == 3
    assert_almost_equal(f(3 - eps), 4, decimal=4)
    assert f(3) == 4
    assert_almost_equal(f(3 + eps), 4, decimal=4)
    assert f(4) == 2.5
    assert_almost_equal(f(5 - eps), 1, decimal=4)
    assert f(5) == 1
    assert f(5 + eps) == 1
    assert f(6) == 1


def test_math_PiecewiseLinear_kwargs():
    from returnn.util.math import PiecewiseLinear

    f = PiecewiseLinear({1: 2, 3: 4, 5: 1}, kw_name="epoch_continuous")
    try:
        f(0)
    except TypeError:
        pass  # this is expected
    else:
        assert False, "TypeError expected (wrong args)"
    assert f(epoch_continuous=0) == 2
    try:
        f(epoch_continuous=0, seq_idx=123)
    except TypeError:
        pass  # this is expected
    else:
        assert False, "TypeError expected (wrong args)"

    f = PiecewiseLinear({1: 2, 3: 4, 5: 1}, kw_name="epoch_continuous", ignore_other_kwargs=True)
    try:
        f(0)
    except TypeError:
        pass  # this is expected
    else:
        assert False, "TypeError expected (wrong args)"
    assert f(epoch_continuous=0) == 2
    assert f(epoch_continuous=0, seq_idx=123) == 2


def test_math_StepFunction_kwargs():
    from returnn.util.math import StepFunction

    f = StepFunction({1: "yes", 3: "no"}, kw_name="epoch", ignore_other_kwargs=True)
    assert f(epoch=0) == "yes"
    assert f(epoch=1) == "yes"
    assert f(epoch=2) == "yes"  # boundary, assume left wins
    assert f(epoch=2.1) == "no"
    assert f(epoch=3) == "no"
    assert f(epoch=100, seq_idx=123) == "no"


def test_parse_orthography_into_symbols():
    assert list("hi") == parse_orthography_into_symbols("hi")
    assert list(" hello ") == parse_orthography_into_symbols(" hello ")
    assert list("  ") == parse_orthography_into_symbols("  ")
    assert list("hello ") + ["[FOO]"] + list(" bar ") == parse_orthography_into_symbols("hello [FOO] bar ")


def test_parse_orthography():
    assert list("hi ") + ["[HES]"] + list(" there") + ["[END]"] == parse_orthography("hi [HES] there ")


def test_NumbersDict_minus_1():
    a = NumbersDict({"classes": 11, "data": 11})
    b = NumbersDict(10)
    r = a - b
    print(a, b, r)
    assert r == NumbersDict(numbers_dict={"classes": 1, "data": 1}, broadcast_value=-10)


def test_NumbersDict_eq_1():
    a = NumbersDict({"classes": 11, "data": 11})
    b = NumbersDict(11)
    r1 = a.elem_eq(b, result_with_default=False)
    r2 = a.elem_eq(b, result_with_default=True)
    r2a = a == b
    print(a, b, r1, r2, r2a)
    assert all(r2.values()) is r2a
    assert r1.value is None
    assert r1.dict == {"classes": True, "data": True}
    assert r1 == NumbersDict({"classes": True, "data": True})
    assert r2.value is None
    assert r2.dict == {"classes": True, "data": True}
    assert r2a is True


def test_NumbersDict_eq_2():
    a = NumbersDict(10)
    assert a == 10
    assert a != 5


def test_NumbersDict_mul():
    a = NumbersDict(numbers_dict={"data": 3, "classes": 2}, broadcast_value=1)
    b = a * 2
    assert isinstance(b, NumbersDict)
    assert b.value == 2
    assert b.dict == {"data": 6, "classes": 4}


def test_NumbersDict_float_div():
    a = NumbersDict(numbers_dict={"data": 3.0, "classes": 2.0}, broadcast_value=1.0)
    b = a / 2.0
    assert isinstance(b, NumbersDict)
    assert_almost_equal(b.value, 0.5)
    assert list(sorted(b.dict.keys())) == ["classes", "data"]
    assert_almost_equal(b.dict["data"], 1.5)
    assert_almost_equal(b.dict["classes"], 1.0)


def test_NumbersDict_int_floordiv():
    a = NumbersDict(numbers_dict={"data": 3, "classes": 2}, broadcast_value=1)
    b = a // 2
    assert isinstance(b, NumbersDict)
    assert b.value == 0
    assert list(sorted(b.dict.keys())) == ["classes", "data"]
    assert b.dict["data"] == 1
    assert b.dict["classes"] == 1


def test_NumbersDict_to_dict():
    a = NumbersDict(numbers_dict={"data": 3, "classes": 2}, broadcast_value=1)
    b = dict(a // 1)
    print(b)
    # We intentionally test whether the order is the same as before.
    assert list(b.keys()) == ["data", "classes"]


def test_collect_class_init_kwargs():
    class A(object):
        def __init__(self, a):
            pass

    class B(A):
        def __init__(self, b, **kwargs):
            super(B, self).__init__(**kwargs)
            pass

    class C(B):
        def __init__(self, b, c, **kwargs):
            super(C, self).__init__(**kwargs)
            pass

    kwargs = collect_class_init_kwargs(C)
    print(kwargs)
    assert sorted(kwargs) == ["a", "b", "c"]


def test_terminal_size():
    terminal_size()


def test_try_get_caller_name():
    def sub():
        return try_get_caller_name()

    assert sub() == "test_try_get_caller_name"


def test_camel_case_to_snake_case():
    assert camel_case_to_snake_case("CamelCaseOp") == "camel_case_op"


def test_NativeCodeCompiler():
    native = NativeCodeCompiler(
        base_name="test_NativeCodeCompiler",
        code_version=1,
        code=textwrap.dedent(
            """\
            static int magic = 13;

            extern "C" void set_magic(int i) { magic = i; }
            extern "C" int get_magic() { return magic; }
            """
        ),
    )
    import ctypes

    lib = native.load_lib_ctypes()
    lib.set_magic.restype = None  # void
    lib.set_magic.argtypes = (ctypes.c_int,)
    lib.get_magic.restype = ctypes.c_int
    lib.get_magic.argtypes = ()

    assert lib.get_magic() == 13
    lib.set_magic(42)
    assert lib.get_magic() == 42


def test_PyExtModCompiler():
    mod_name = "TestPyExtModCompiler"
    native = PyExtModCompiler(
        base_name=mod_name,
        verbose=True,
        code_version=1,
        code=textwrap.dedent(
            f"""\
            #include <Python.h>

            static PyObject *
            func(PyObject *self, PyObject *args)
            {{
                int x;
                if (!PyArg_ParseTuple(args, "i", &x))
                    return NULL;
                return PyLong_FromLong(x + 42);
            }}

            static PyMethodDef _Methods[] = {{
                {{"func",  func, METH_VARARGS, "Func."}},
                {{NULL, NULL, 0, NULL}}        /* Sentinel */
            }};

            static struct PyModuleDef _Module = {{
                PyModuleDef_HEAD_INIT,
                "{mod_name}",   /* name of module */
                NULL,     /* module documentation, may be NULL */
                -1,       /* size of per-interpreter state of the module,
                             or -1 if the module keeps state in global variables. */
                _Methods
            }};

            PyMODINIT_FUNC
            PyInit_{mod_name}(void)
            {{
                return PyModule_Create(&_Module);
            }}
            """
        ),
    )
    lib_filename = native.get_lib_filename()
    print("lib_filename:", lib_filename)
    mod = native.load_py_module()

    assert mod.func(0) == 42
    assert mod.func(1) == 43
    assert mod.func(2) == 44
    assert mod.func(-2) == 40


def test_Stats():
    rnd = numpy.random.RandomState(42)
    m = rnd.uniform(-2.0, 10.0, (1000, 3))
    mean_ref = numpy.mean(m, axis=0)
    var_ref = numpy.var(m, axis=0)
    std_dev_ref = numpy.std(m, axis=0)
    print("ref mean/var/stddev:", mean_ref, var_ref, std_dev_ref)
    assert_almost_equal(numpy.sqrt(var_ref), std_dev_ref)
    stats = Stats()
    t = 0
    while t < len(m):
        s = int(rnd.uniform(10, 100))
        m_sub = m[t : t + s]
        print("sub seq from t=%i, len=%i" % (t, len(m_sub)))
        stats.collect(m_sub)
        t += s
    mean = stats.get_mean()
    std_dev = stats.get_std_dev()
    print("mean/stddev:", mean, std_dev)
    assert_almost_equal(mean, mean_ref)
    assert_almost_equal(std_dev, std_dev_ref)
    m -= mean[None, :]
    m /= std_dev[None, :]
    stats2 = Stats()
    t = 0
    while t < len(m):
        s = int(rnd.uniform(10, 100))
        m_sub = m[t : t + s]
        stats2.collect(m_sub)
        t += s
    mean0 = stats2.get_mean()
    stddev1 = stats2.get_std_dev()
    print("normalized mean/stddev:", mean0, stddev1)
    assert_almost_equal(mean0, 0.0)
    assert_almost_equal(stddev1, 1.0)


def test_deepcopy():
    deepcopy({"a": 1, "b": 2, "c": [3, {}, (), [42, True]]})


def test_deepcopy_mod():
    o = deepcopy({"a": 1, "b": 2, "c": sys})
    assert isinstance(o, dict)
    assert o["c"] is sys


def test_deepcopy_config():
    from returnn.config import Config

    config = Config()
    deepcopy(config)


def test_deepcopy_builtins():
    user_ns = {}
    custom_exec("", "<source.py>", user_ns, user_ns)
    print(user_ns)
    assert "__builtins__" in user_ns
    assert isinstance(user_ns["__builtins__"], dict)
    o = deepcopy(user_ns)
    assert o["__builtins__"] is user_ns["__builtins__"]  # no copy, directly reference this module dict


_demo_entity = Entity("demo entity", global_name="_demo_entity")


def test_deepcopy_Entity():
    entity_copy = deepcopy(_demo_entity)
    assert entity_copy is _demo_entity  # no copy, directly reference this entity


def test_get_func_kwargs():
    def dummy_func(net, var, update_ops):
        pass

    assert list(getargspec(dummy_func).args) == ["net", "var", "update_ops"]


def test_next_type_attrib_in_mro_chain():
    class Base:
        def method(self):
            return id(self)

    class Foo(Base):
        def method(self):
            return 2

    class Bar(Foo):
        pass

    assert type_attrib_mro_chain(Foo, "method") == [Foo.method, Base.method]
    assert type_attrib_mro_chain(Bar, "method") == [Foo.method, Base.method]
    assert next_type_attrib_in_mro_chain(Bar, "method", Foo.method) == Base.method


def test_simple_obj_repr():
    class X:
        def __init__(self, a, b=13):
            self.a = a
            self.b = b

        __repr__ = simple_obj_repr

    x = X(a=42)
    x_repr = repr(x)

    assert x_repr == "X(a=42, b=13)"


def test_obj_diff_str():
    assert obj_diff_str({"a": 1, "b": 2}, {"a": 1, "b": 3}) == "dict diff:\n['b'] self: 2 != other: 3"


def test_obj_diff_str_non_str_key():
    assert obj_diff_str({1: 1, 2: 2}, {1: 1, 2: 3}) == "dict diff:\n[2] self: 2 != other: 3"


def test_obj_diff_list_allowed_mapping():
    a = {"a": 1, "b": {"A:a": 1, "A:b": 2}, "c": ["A:a"], "d": {"A:b"}}
    b = {"a": 1, "b": {"B:a": 1, "B:b": 2}, "c": ["B:a"], "d": {"B:b"}}
    c = {"a": 1, "b": {"B:a": 1, "B:b": 2}, "c": ["B:a"], "d": {"B:x"}}
    ab_diff = obj_diff_list(a, b)
    assert ab_diff
    for line in ab_diff:
        print(line)
    print("---")

    def _allowed_mapping(a_, b_):
        if isinstance(a_, str) and isinstance(b_, str):
            if a_.startswith("A:") and b_.startswith("B:"):
                return True
        return False

    ab_diff = obj_diff_list(a, b, allowed_mapping=_allowed_mapping)
    assert not ab_diff
    ac_diff = obj_diff_list(a, c, allowed_mapping=_allowed_mapping)
    assert ac_diff
    for line in ac_diff:
        print(line)
    assert ac_diff == [
        "dict diff:",
        "['b'] dict diff:",
        "['b']   key 'A:b' not in other",
        "['b']   key 'B:b' not in self",
    ]


def test_CollectionReadCheckCovered():
    x = CollectionReadCheckCovered.from_bool_or_dict(True)
    assert x and x.truth_value
    x = CollectionReadCheckCovered.from_bool_or_dict(False)
    assert not x or not x.truth_value
    x = CollectionReadCheckCovered.from_bool_or_dict({})
    assert not x or not x.truth_value
    x = CollectionReadCheckCovered.from_bool_or_dict({"a": "b"})
    assert x and x.truth_value


def test_version():
    import returnn

    print("RETURNN version:", returnn.__version__, returnn.__long_version__)
    assert "1.0.0" not in returnn.__version__ and "unknown" not in returnn.__long_version__


def test_logging():
    # There is no real test. But you can interactively test this.
    import logging

    logging.getLogger("returnn").debug("Hello from returnn logger.")
    logging.getLogger("returnn.import_").debug("Hello from returnn.import_ logger.")


def test_import_():
    from returnn.import_ import import_

    mod = import_("github.com/rwth-i6/returnn-experiments", "common/test.py", "20210302-01094bef2761")
    print("Loaded mod %s, name %s, file %s" % (mod, mod.__name__, mod.__file__))
    assert mod.hello() == "hello world"


def test_import_root_repo_mod():
    from returnn.import_ import import_

    mod = import_("github.com/rwth-i6/returnn_common", "test.py", "20210602-1bc6822")
    print("Loaded mod %s, name %s, file %s" % (mod, mod.__name__, mod.__file__))
    assert mod.hello() == "hello world"


def test_import_root_repo_pkg():
    from returnn.import_ import import_

    mod = import_("github.com/rwth-i6/returnn_common", ".", "20210602-1bc6822")
    print("Loaded mod %s, name %s, file %s" % (mod, mod.__name__, mod.__file__))
    from returnn_import.github_com.rwth_i6.returnn_common.v20210602162042_1bc6822b2fd1 import test

    assert test.hello() == "hello world"


def test_import_root_repo_sub_mod():
    from returnn.import_ import import_

    mod = import_("github.com/rwth-i6/returnn_common", "test/hello.py", "20210603-3752d77")
    print("Loaded mod %s, name %s, file %s" % (mod, mod.__name__, mod.__file__))
    assert mod.hello() == "hello world"


def test_import_pkg_py_import():
    from returnn.import_ import import_

    mod = import_("github.com/rwth-i6/returnn-experiments", "common", "20210302-01094bef2761")
    print("Loaded mod %s, name %s, file %s" % (mod, mod.__name__, mod.__file__))
    # noinspection PyUnresolvedReferences
    from returnn_import.github_com.rwth_i6.returnn_experiments.v20210302133012_01094bef2761 import common

    assert common is mod


def test_import_wrong_date():
    from returnn.import_ import import_
    from returnn.import_.common import InvalidVersion

    try:
        import_("github.com/rwth-i6/returnn-experiments", "common/test.py", "20210301-01094bef2761")
    except InvalidVersion as exc:
        print("got expected exception:", exc)
    else:
        raise Exception("We expected an invalid version exception but got nothing.")


def test_import_wrong_pkg_py_import():
    from pprint import pprint
    from returnn.import_ import import_
    from returnn.import_.common import _registered_modules, MissingExplicitImport

    # Use some other commit here which is not used by the other tests, to not mess up.
    mod = import_("github.com/rwth-i6/returnn-experiments", "common", "20210302-10beb9d1c57e")
    print("Loaded mod %s, name %s, file %s" % (mod, mod.__name__, mod.__file__))
    print("Registered modules:")
    pprint(_registered_modules)

    # Mod name, without "common".
    mod_name = "returnn_import.github_com.rwth_i6.returnn_experiments.v20210302153450_10beb9d1c57e"
    assert mod_name in _registered_modules
    assert mod_name in sys.modules
    # Remove from both to force a reload.
    del _registered_modules[mod_name]
    del sys.modules[mod_name]

    try:
        import returnn_import.github_com.rwth_i6.returnn_experiments.v20210302153450_10beb9d1c57e
    except MissingExplicitImport as exc:  # but *not* a normal ImportError
        print("Got expected exception:", exc)
    else:
        raise Exception("We expected an import error exception but got nothing.")


def test_literal_py_to_pickle():
    import ast
    from returnn.util import literal_py_to_pickle

    def check(s):
        print("check:", s)
        a = ast.literal_eval(s)
        b = literal_py_to_pickle.literal_eval(s)
        assert a == b

    checks = [
        "0",
        "1",
        "0.0",
        "1.23",
        '""',
        '"abc"',
        "''",
        "'abc'",
        '"abc\\n\\x00\\x01\\"\'abc"',
        "'hällö'",
        "'k\\u200eý'",
        "[]",
        "[1]",
        "[1,2,3]",
        "{}",
        "{'a': 'b', 1: 2}",
        "{1}",
        "{1,2,3}",
        "None",
        "{'a': None, 'b':1, 'c':None, 'd':'d'}",
    ]
    for s in checks:
        check(s)


def test_native_signal_handler():
    from returnn.util.debug import install_native_signal_handler

    install_native_signal_handler(reraise_exceptions=True)


class _PreInitUnpicklingRaiseException:
    def __getstate__(self):
        return 42

    def __setstate__(self, state):
        raise Exception("_PreInitUnpicklingRaiseException")


class _PreInitDummy:
    def __init__(self, *, payload):
        self.payload = payload

    def __call__(self):
        pass  # nothing


def _noop_func():
    pass


def test_NonDaemonicSpawnProcess_hang_1514():
    # https://github.com/rwth-i6/returnn/issues/1514
    from returnn.util.multi_proc_non_daemonic_spawn import NonDaemonicSpawnProcess

    with timeout():
        proc = NonDaemonicSpawnProcess(target=_noop_func)
        proc.start()
        proc.join()
        assert proc.exitcode == 0

        proc = NonDaemonicSpawnProcess(target=_noop_func)
        # Give it some payload large enough such that it will the pipe buffer to trigger the potential hang.
        proc.pre_init_func = _PreInitDummy(payload=["A" * 100_000, _PreInitUnpicklingRaiseException(), "B" * 100_000])
        # After fixing this, we expect that the unpickling of pre_init_func will raise an exception,
        # however, the parent then will not hang.
        proc.start()
        proc.join()
        assert proc.exitcode != 0


def test_expand_env_vars():
    os.environ["TMPA"] = "/testA"
    os.environ["TMPB"] = "testB"
    assert expand_env_vars("$TMPA/$TMPB/returnn/file_cache") == "/testA/testB/returnn/file_cache"


def test_bpe_PrefixTree():
    from returnn.util.bpe import PrefixTree, BpeOpts, BpePostMergeSymbol

    tree = PrefixTree(opts=BpeOpts(label_postfix_merge_symbol=BpePostMergeSymbol))
    tree.add("hello")
    tree.add("helo" + BpePostMergeSymbol)
    assert not tree.finished and not tree.bpe_finished
    assert set(tree.arcs.keys()) == {"h"}
    node = tree.arcs["h"]
    assert not node.finished and not node.bpe_finished
    assert set(node.arcs.keys()) == {"e"}
    node = node.arcs["e"]
    assert not node.finished and not node.bpe_finished
    assert set(node.arcs.keys()) == {"l"}
    node = node.arcs["l"]
    assert not node.finished and not node.bpe_finished
    assert set(node.arcs.keys()) == {"l", "o"}
    node_o = node.arcs["o"]
    assert not node_o.finished and node_o.bpe_finished
    # The following BpePostMergeSymbol arc and its node is somewhat arbitrary, probably does not matter...
    assert set(node_o.arcs.keys()) == {BpePostMergeSymbol}
    node_o_post = node_o.arcs[BpePostMergeSymbol]
    assert node_o_post.finished and not node_o_post.bpe_finished
    assert set(node_o_post.arcs.keys()) == set()


def test_bpe_PrefixTree_word_prefix():
    from returnn.util.bpe import PrefixTree, BpeOpts

    tree = PrefixTree(opts=BpeOpts(word_prefix_symbol="▁"))
    tree.add("▁hello")
    tree.add("▁hel")
    tree.add("lo")
    tree.add("▁hi")
    assert not tree.finished and not tree.bpe_finished
    assert set(tree.arcs.keys()) == {"▁", "l"}
    node = tree.arcs["▁"]
    assert not node.finished and not node.bpe_finished
    assert set(node.arcs.keys()) == {"h"}
    node = node.arcs["h"]
    assert not node.finished and not node.bpe_finished
    assert set(node.arcs.keys()) == {"e", "i"}
    node_ = node.arcs["i"]
    assert node_.finished and not node_.bpe_finished and not node_.arcs
    node = node.arcs["e"]
    assert not node.finished and not node.bpe_finished
    assert set(node.arcs.keys()) == {"l"}
    node = node.arcs["l"]
    assert node.finished and not node.bpe_finished
    assert set(node.arcs.keys()) == {"l"}
    node = node.arcs["l"]
    assert not node.finished and not node.bpe_finished
    assert set(node.arcs.keys()) == {"o"}
    node = node.arcs["o"]
    assert node.finished and not node.bpe_finished and not node.arcs


def test_bpe_DepthFirstSearch():
    import itertools
    from returnn.util.bpe import PrefixTree, BpeOpts, DepthFirstSearch

    tree = PrefixTree(opts=BpeOpts(label_postfix_merge_symbol="@@"))
    tree.add("llo")
    tree.add("helo@@")
    tree.add("he@@")

    dfs = DepthFirstSearch(tree, "hello")
    assert dfs.search() == ["he@@", "llo"]
    dfs = DepthFirstSearch(tree, "helo")
    assert dfs.search() == None
    dfs = DepthFirstSearch(tree, "x")
    assert dfs.search() == None
    dfs = DepthFirstSearch(tree, "llo")
    assert dfs.search() == ["llo"]

    tree.add("hello")
    dfs = DepthFirstSearch(tree, "hello")
    assert dfs.search() == ["hello"]
    dfs = DepthFirstSearch(tree, "hello", sampler=lambda: True)
    assert dfs.search() == ["he@@", "llo"]

    tree.add("hel@@")
    tree.add("lo")
    dfs = DepthFirstSearch(tree, "hello")
    assert dfs.search() == ["hello"]
    dfs = DepthFirstSearch(tree, "hello", sampler=lambda: True)
    assert dfs.search() == ["he@@", "llo"]
    dfs = DepthFirstSearch(tree, "hello", sampler=lambda _it=itertools.count(): next(_it) in {3})
    assert dfs.search() == ["hel@@", "lo"]


def test_bpe_DepthFirstSearch_word_prefix():
    import itertools
    from returnn.util.bpe import PrefixTree, BpeOpts, DepthFirstSearch

    tree = PrefixTree(opts=BpeOpts(word_prefix_symbol="▁"))
    tree.add("▁hello")
    tree.add("▁hel")
    tree.add("lo")

    search = DepthFirstSearch(tree, "hello")
    assert search.search() == ["▁hello"]
    search = DepthFirstSearch(tree, "hello", sampler=lambda: True)
    assert search.search() == ["▁hel", "lo"]

    tree.add("▁he")
    tree.add("llo")
    search = DepthFirstSearch(tree, "hello", sampler=lambda: True)
    assert search.search() == ["▁he", "llo"]
    search = DepthFirstSearch(tree, "hello", sampler=lambda _it=itertools.count(): next(_it) in {3})
    assert search.search() == ["▁hel", "lo"]


def test_bpe_CharSyncSearch():
    from returnn.util.bpe import PrefixTree, BpeOpts, CharSyncSearch

    tree = PrefixTree(opts=BpeOpts(label_postfix_merge_symbol="@@"))
    tree.add("llo")
    tree.add("helo@@")
    tree.add("he@@")

    search = CharSyncSearch(tree, "hello")
    assert search.search() == [["he@@", "llo"]]
    search = CharSyncSearch(tree, "helo")
    assert search.search() == []
    search = CharSyncSearch(tree, "x")
    assert search.search() == []
    search = CharSyncSearch(tree, "llo")
    assert search.search() == [["llo"]]

    tree.add("hello")
    search = CharSyncSearch(tree, "hello")
    assert search.search() == [["he@@", "llo"], ["hello"]]


def test_bpe_CharSyncSearch_word_prefix():
    from returnn.util.bpe import PrefixTree, BpeOpts, CharSyncSearch

    tree = PrefixTree(opts=BpeOpts(word_prefix_symbol="▁"))
    tree.add("▁hello")
    tree.add("▁hel")
    tree.add("lo")
    tree.add("▁hi")

    search = CharSyncSearch(tree, "hello")
    assert search.search() == [["▁hel", "lo"], ["▁hello"]]
    search = CharSyncSearch(tree, "helo")
    assert search.search() == []
    search = CharSyncSearch(tree, "x")
    assert search.search() == []
    search = CharSyncSearch(tree, "lo")
    assert search.search() == []
    search = CharSyncSearch(tree, "hi")
    assert search.search() == [["▁hi"]]


def test_file_cache():
    from returnn.util.file_cache import FileCache, CachedFile

    # First create some dummy data, to be used for the test below.
    src_dir = _get_tmp_dir() + "/returnn/file_cache_src"
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(src_dir + "/dirA/subdirB", exist_ok=True)
    os.makedirs(src_dir + "/dirC", exist_ok=True)
    with open(src_dir + "/dummy1.txt", "w") as f:
        f.write("Hello dummy1.txt\n")
    with open(src_dir + "/dirA/subdirB/dummy2.txt", "w") as f:
        f.write("Hello dummy2.txt\n")
    with open(src_dir + "/dirC/dummy3.txt", "w") as f:
        f.write("Hello dummy3.txt\n")

    cache_dir = _get_tmp_dir() + "/returnn/file_cache"
    cache = FileCache(cache_directory=cache_dir)
    cache._lock_timeout = 5  # for testing

    # Copy some dummy file.
    cached_fn1 = cache.get_file(src_dir + "/dummy1.txt")
    assert cached_fn1 == cache_dir + src_dir + "/dummy1.txt" and os.path.exists(cached_fn1)
    with open(cached_fn1) as f:
        assert f.read() == "Hello dummy1.txt\n"
    cached_fn2 = cache.get_file(src_dir + "/dirA/subdirB/dummy2.txt")
    assert cached_fn2 == cache_dir + src_dir + "/dirA/subdirB/dummy2.txt" and os.path.exists(cached_fn2)
    with open(cached_fn2) as f:
        assert f.read() == "Hello dummy2.txt\n"
    cached_fn3 = cache.get_file(src_dir + "/dirC/dummy3.txt")
    assert cached_fn3 == cache_dir + src_dir + "/dirC/dummy3.txt" and os.path.exists(cached_fn3)
    with open(cached_fn3) as f:
        assert f.read() == "Hello dummy3.txt\n"
    target_values = {
        cache_dir + src_dir + "/dummy1.txt": 1,
        cache_dir + src_dir + "/dirA/subdirB/dummy2.txt": 1,
        cache_dir + src_dir + "/dirC/dummy3.txt": 1,
    }
    for k, v in cache._touch_files_thread.files.items():
        assert FileCache._is_info_filename(k) or (k in target_values and target_values[k] == v)

    # Check config handle_cached_files_in_config.
    config, config_cached_files = cache.handle_cached_files_in_config(
        {"class": "Dataset", "files": [CachedFile(src_dir + "/dirA/subdirB/dummy2.txt")]}
    )
    assert config == {"class": "Dataset", "files": [cache_dir + src_dir + "/dirA/subdirB/dummy2.txt"]}
    assert config_cached_files == [cache_dir + src_dir + "/dirA/subdirB/dummy2.txt"]

    target_values = {
        cache_dir + src_dir + "/dummy1.txt": 1,
        cache_dir + src_dir + "/dirA/subdirB/dummy2.txt": 2,
        cache_dir + src_dir + "/dirC/dummy3.txt": 1,
    }
    for k, v in cache._touch_files_thread.files.items():
        assert FileCache._is_info_filename(k) or (k in target_values and target_values[k] == v)

    # Check that file is kept up-to-date until we release it.
    mtimes = set()
    for t in range(10):
        print(f"Sec {t}...")
        mtime = os.stat(cache_dir + src_dir + "/dirA/subdirB/dummy2.txt").st_mtime
        mtimes.add(mtime)
        assert 0 <= time.time() - mtime < 2
        time.sleep(1)
    assert len(mtimes) > 1, "mtime should change"

    # Now release the files.
    cache.release_files(
        [
            cache_dir + src_dir + "/dummy1.txt",
            cache_dir + src_dir + "/dirA/subdirB/dummy2.txt",
            cache_dir + src_dir + "/dirC/dummy3.txt",
        ]
    )
    cache.release_files([cache_dir + src_dir + "/dirA/subdirB/dummy2.txt"])  # was acquired twice
    assert len(dict(cache._touch_files_thread.files)) == 0
    time.sleep(5)
    mtime = os.stat(cache_dir + src_dir + "/dirA/subdirB/dummy2.txt").st_mtime
    assert 4 <= time.time() - mtime

    # Check cleanup mechanism.
    cache._cleanup_files_always_older_than_days = 4 / 60 / 60 / 24
    print("Cache older than secs:", cache._cleanup_files_always_older_than_days * 24 * 60 * 60)
    cleanup_timestamp_file = cache_dir + "/.recent_full_cleanup"
    try:
        last_full_cleanup = os.stat(cleanup_timestamp_file).st_mtime
    except FileNotFoundError:
        last_full_cleanup = float("-inf")
    print("Recent cleanup is ago:", time.time() - last_full_cleanup, "secs")
    # Note: After the first cleanup, there are still dirs maybe left over, because dirs are only cleaned
    # when their mtime is a older than some threshold.
    # However, when we just deleted some file inside, that would have updated the mtime of the dir.
    # This effectively means that we need several cleanup runs to clean up also the empty dirs.
    cache._lock_timeout = 0.05  # used for the threshold
    prev_count_dirs = None
    while True:
        time.sleep(0.1)
        os.unlink(cleanup_timestamp_file)  # reset
        time.sleep(0.1)
        cache.cleanup()
        count_files = 0
        count_dirs = 0
        for root, dirs, files in os.walk(cache_dir):
            # ignore cleanup timestamp file
            for fn in dirs + files:
                print(f"{root}/{fn}", "age:", time.time() - os.stat(f"{root}/{fn}").st_mtime, "sec")
            leftover_files = [f for f in files if "recent_full_cleanup" not in f]
            count_files += len(leftover_files)
            count_dirs += len(dirs)
        print("count files:", count_files, "count dirs:", count_dirs)
        assert count_files == 0
        if prev_count_dirs is None:
            assert count_dirs > 0
        else:
            assert count_dirs < prev_count_dirs
        prev_count_dirs = count_dirs
        if count_dirs == 0:
            break


if __name__ == "__main__":
    better_exchook.install()
    if len(sys.argv) <= 1:
        for k, v in sorted(globals().items()):
            if k.startswith("test_"):
                print("-" * 40)
                print("Executing: %s" % k)
                try:
                    v()
                except unittest.SkipTest as exc:
                    print("SkipTest:", exc)
                print("-" * 40)
        print("Finished all tests.")
    else:
        assert len(sys.argv) >= 2
        for arg in sys.argv[1:]:
            print("Executing: %s" % arg)
            if arg in globals():
                globals()[arg]()  # assume function and execute
            else:
                eval(arg)  # assume Python code and execute
