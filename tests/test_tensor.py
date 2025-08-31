"""
tests for returnn.tensor
"""

import _setup_test_env  # noqa
import sys
import unittest
from returnn.util import better_exchook
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim


def test_tensor():
    batch_dim = Dim(name="batch", dimension=None)
    time_dim = Dim(name="time", dimension=None)
    feat_dim = Dim(10)
    x = Tensor("x", (batch_dim, time_dim, feat_dim), "float32")
    print(x)


def test_dim_math_pad_window():
    rf.select_backend("numpy")
    batch_dim = Dim(3, name="batch")
    time1_dim = Dim(rf.convert_to_tensor([5, 6, 7], dims=[batch_dim], name="time1"))
    time2_dim = Dim(rf.convert_to_tensor([20, 21, 22], dims=[batch_dim], name="time2"))
    time3_dim = Dim(rf.convert_to_tensor([10, 5, 3], dims=[batch_dim], name="time3"))
    in_spatial_dim = time1_dim + time2_dim + time3_dim
    filter_size = 17
    # As it would happen with conv/pool/window/... with window size 17.
    out_spatial_dim = in_spatial_dim.sub_left(8).sub_right(8)
    print("out_spatial_dim:", out_spatial_dim)
    sizes = out_spatial_dim.dyn_size.tolist()
    print("sizes:", sizes)
    expected_sizes = (time1_dim.dyn_size + time2_dim.dyn_size + time3_dim.dyn_size - (filter_size - 1)).tolist()
    print("expected_sizes:", expected_sizes)
    assert sizes == expected_sizes


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
