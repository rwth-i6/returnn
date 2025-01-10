"""
test threading-specific functionality
"""

from __future__ import annotations

import _setup_test_env  # noqa
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
import os
import sys
import unittest
import tempfile
import textwrap
from returnn.util import better_exchook


__my_dir__ = os.path.dirname(os.path.abspath(__file__))
__base_dir__ = os.path.dirname(__my_dir__)
__main_entry__ = __base_dir__ + "/rnn.py"
py = sys.executable


def run(args, input=None):
    """run subproc"""
    args = list(args)
    print("run:", args)
    # RETURNN by default outputs on stderr, so just merge both together
    p = Popen(args, stdout=PIPE, stderr=STDOUT, stdin=PIPE)
    out, _ = p.communicate(input=input)
    print("Return code is %i" % p.returncode)
    print("std out/err:\n---\n%s\n---\n" % out.decode("utf8"))
    if p.returncode != 0:
        raise CalledProcessError(cmd=args, returncode=p.returncode, output=out)
    return out.decode("utf8")


def test_thread_exc_hook():
    config = textwrap.dedent(
        """
        #!rnn.py

        backend = "torch"  # just require any backend
        log_verbosity = 5
        task = "nop"

        import threading

        t = threading.Thread(target=lambda: 1/0)
        t.start()
        t.join()
        """
    )
    with tempfile.NamedTemporaryFile(mode="w") as f:
        f.write(config)
        f.flush()

        try:
            output = run([py, __main_entry__, f.name])
        except CalledProcessError as exc:
            out = exc.output.decode("utf8")
            assert "ZeroDivisionError" in out
            assert "KeyboardInterrupt" in out  # this is the result from the main thread being killed
        else:
            assert False, f"Expected RETURNN to crash, but got {output}."


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
    #  better_exchook.dump_all_thread_tracebacks()
