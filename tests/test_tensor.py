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


# Note: Some of the dim math tests are duplicated from test_TFUtil.py, and partly adapted.
# But here they are TF-independent.


def test_dim_math_basics():
    a = Dim(None, name="a")
    b = Dim(None, name="b")
    assert a == a
    assert (a + 2 - 2) == a
    assert a + b == a + b
    assert a + b != b + a  # not commutative
    assert a * b == a * b
    assert a * b != b * a  # not commutative
    assert 2 * a == a + a
    assert 3 * a == a + a + a
    assert a * 2 != 2 * a
    assert 2 * a + b == a + a + b
    assert a + b - b == a
    assert a + 2 * b - b + -b == a
    assert a * b + b == (a + 1) * b
    assert (a + b) * 2 == a * 2 + b * 2
    assert 0 + a + 0 == a
    assert sum([0, a, 0, a, 0]) == 2 * a


def test_dim_math_double_neg():
    a = Dim(None, name="a")
    assert --a == a


def test_dim_math_mul_div():
    a = Dim(None, name="a")
    b = Dim(None, name="b")
    assert (a * b) // b == a
    assert (b * a) // b != a
    assert (b * a).div_left(b) == a


def test_dim_math_div_mul():
    a = Dim(None, name="a")
    b = Dim(None, name="b")
    assert a // b == a // b


def test_dim_math_div_div():
    a = Dim(None, name="a")
    b = a.ceildiv_right(2)
    b = b.ceildiv_right(3)
    c = a.ceildiv_right(6)
    print(a, b, c)
    assert b == c


def test_dim_math_pad_conv():
    time = Dim(None, name="time")
    padded = 2 + time + 2
    assert padded == 2 + time + 2
    conv_valid = (-2) + padded + (-2)
    assert conv_valid == time


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
