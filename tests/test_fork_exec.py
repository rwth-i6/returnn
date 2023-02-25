"""
What we want: subprocess.Popen to always work.
Problem: It uses fork+exec internally in subprocess_fork_exec, via _posixsubprocess.fork_exec.
That is a problem because fork can trigger any atfork handlers registered via pthread_atfork,
and those can crash/deadlock in some cases.

https://github.com/tensorflow/tensorflow/issues/13802
https://github.com/xianyi/OpenBLAS/issues/240
https://trac.sagemath.org/ticket/22021
https://bugs.python.org/issue31814

"""

from __future__ import annotations

import _setup_test_env  # noqa
import os
import sys
from nose.tools import assert_equal, assert_is_instance
from pprint import pprint


my_dir = os.path.dirname(os.path.abspath(__file__))

c_code_at_fork_demo = """
#include <stdio.h>
#include <pthread.h>
// int pthread_atfork(void (*prepare)(void), void (*parent)(void), void (*child)(void));

static long magic_number = 0;

void set_magic_number(long i) {
    magic_number = i;
}

void hello_from_child() {
    printf("Hello from child atfork, magic number %li.\\n", magic_number);
    fflush(stdout);
}

void hello_from_fork_prepare() {
    printf("Hello from atfork prepare, magic number %li.\\n", magic_number);
    fflush(stdout);
}

void register_hello_from_child() {
    pthread_atfork(0, 0, &hello_from_child);
}

void register_hello_from_fork_prepare() {
    pthread_atfork(&hello_from_fork_prepare, 0, 0);
}
"""


class CLibAtForkDemo:
    def __init__(self):
        self._load_lib()

    def _load_lib(self):
        from returnn.util.basic import NativeCodeCompiler

        native = NativeCodeCompiler(
            base_name="test_fork_exec", code_version=1, code=c_code_at_fork_demo, is_cpp=False, ld_flags=["-lpthread"]
        )
        self._lib = native.load_lib_ctypes()
        print("loaded lib:", native.get_lib_filename())
        import ctypes

        self._lib.register_hello_from_child.restype = None  # void
        self._lib.register_hello_from_child.argtypes = ()
        self._lib.register_hello_from_fork_prepare.restype = None  # void
        self._lib.register_hello_from_fork_prepare.argtypes = ()
        self._lib.set_magic_number.restype = None  # void
        self._lib.set_magic_number.argtypes = (ctypes.c_long,)

    def set_magic_number(self, i):
        self._lib.set_magic_number(i)

    def register_hello_from_child(self):
        self._lib.register_hello_from_child()

    def register_hello_from_fork_prepare(self):
        self._lib.register_hello_from_fork_prepare()


clib_at_fork_demo = CLibAtForkDemo()


def demo_hello_from_fork():
    print("Hello.")
    sys.stdout.flush()
    clib_at_fork_demo.set_magic_number(3)
    clib_at_fork_demo.register_hello_from_child()
    clib_at_fork_demo.register_hello_from_fork_prepare()
    pid = os.fork()
    if pid == 0:
        print("Hello from child after fork.")
        sys.exit()
    print("Hello from parent after fork.")
    os.waitpid(pid, 0)
    print("Bye.")


def demo_start_subprocess():
    print("Hello.")
    sys.stdout.flush()
    clib_at_fork_demo.set_magic_number(5)
    clib_at_fork_demo.register_hello_from_child()
    clib_at_fork_demo.register_hello_from_fork_prepare()
    from subprocess import check_call

    # Right now (2017-10-19), CPython will use subprocess_fork_exec which
    # uses fork+exec, so this will call the atfork handler.
    check_call("echo Hello from subprocess.", shell=True)
    print("Bye.")


def run_demo_check_output(name):
    """
    :param str name: e.g. "demo_hello_from_fork"
    :return: lines of stdout of the demo
    :rtype: list[str]
    """
    from subprocess import check_output

    output = check_output([sys.executable, __file__, name])
    return output.decode("utf8").splitlines()


def filter_demo_output(ls):
    """
    :param list[str] ls:
    :rtype: list[str]
    """
    ls = [l for l in ls if not l.startswith("Executing: ")]
    ls = [l for l in ls if not l.startswith("Compiler call: ")]
    ls = [l for l in ls if not l.startswith("loaded lib: ")]
    ls = [l for l in ls if not l.startswith("dlopen: ")]
    ls = [l for l in ls if "installLibSigSegfault" not in l and "libSegFault" not in l]
    ls = [l for l in ls if "faulthandler" not in l]
    found_hello = False
    for i, l in enumerate(ls):
        # Those can be very early, before the hello.
        if l in ["Ignoring pthread_atfork call!", "Ignoring __register_atfork call!"]:
            continue
        assert l == "Hello."
        ls.pop(i)
        found_hello = True
        break
    assert found_hello, "no Hello: %r" % (ls,)
    assert ls[-1] == "Bye."
    ls = ls[:-1]
    return ls


def test_demo_hello_from_fork():
    ls = run_demo_check_output("demo_hello_from_fork")
    pprint(ls)
    ls = filter_demo_output(ls)
    pprint(ls)
    assert_equal(
        set(ls),
        {
            "Hello from child after fork.",
            "Hello from child atfork, magic number 3.",
            "Hello from atfork prepare, magic number 3.",
            "Hello from parent after fork.",
        },
    )


def test_demo_start_subprocess():
    ls = run_demo_check_output("demo_start_subprocess")
    pprint(ls)
    ls = filter_demo_output(ls)
    pprint(ls)
    assert "Hello from subprocess." in ls
    # Maybe this is fixed/changed in some later CPython version.
    import platform

    print("Python impl:", platform.python_implementation())
    print("Python version:", sys.version_info[:3])
    if platform.python_implementation() == "CPython":
        if sys.version_info[0] == 2:
            old = sys.version_info[:3] <= (2, 7, 12)
        else:
            old = sys.version_info[:3] <= (3, 6, 1)
    else:
        old = False
    if old:
        print("atfork handler should have been called.")
        assert "Hello from child atfork, magic number 5." in ls
        assert "Hello from atfork prepare, magic number 5." in ls
    else:
        print("Not checking for atfork handler output.")


def patched_check_demo_start_subprocess():
    """
    Just like test_demo_start_subprocess(), but here we assert that no atfork handlers are executed.
    """
    assert_equal(os.environ.get("__RETURNN_ATFORK_PATCHED"), "1")
    ls = run_demo_check_output("demo_start_subprocess")
    pprint(ls)
    ls = filter_demo_output(ls)
    pprint(ls)
    assert "Hello from subprocess." in ls
    ls = [l for l in ls if l not in ["Ignoring pthread_atfork call!", "Ignoring __register_atfork call!"]]
    pprint(ls)
    assert_equal(ls, ["Hello from subprocess."])


def test_demo_start_subprocess_patched():
    from returnn.util.basic import get_patch_atfork_lib
    from subprocess import check_call

    env = os.environ.copy()
    env["LD_PRELOAD"] = get_patch_atfork_lib()
    print("LD_PRELOAD:", get_patch_atfork_lib())
    check_call([sys.executable, __file__, "patched_check_demo_start_subprocess"], env=env)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        for k, v in sorted(globals().items()):
            if k.startswith("test_"):
                print("-" * 40)
                print("Executing: %s" % k)
                v()
                print("-" * 40)
    else:
        assert len(sys.argv) >= 2
        for arg in sys.argv[1:]:
            print("Executing: %s" % arg)
            if arg in globals():
                globals()[arg]()  # assume function and execute
            else:
                eval(arg)  # assume Python code and execute
