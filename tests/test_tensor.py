"""
tests for returnn.tensor
"""

import _setup_test_env  # noqa
import sys
import unittest
from returnn.util import better_exchook
from returnn.tensor import Tensor, Dim


def test_tensor():
    batch_dim = Dim(name="batch", dimension=None)
    time_dim = Dim(name="time", dimension=None)
    feat_dim = Dim(10)
    x = Tensor("x", (batch_dim, time_dim, feat_dim), "float32")
    print(x)


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
