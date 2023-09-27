"""
tests for returnn.torch.frontend
"""

import _setup_test_env  # noqa

import numpy
import sys
import torch
import pytest
import unittest

from returnn.util import better_exchook
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


rf.select_backend_torch()


# Tensor.__eq__ is disabled as per the following error in some TF tests:
# AssertionError: unhashable type: 'Tensor'.
# See CI https://github.com/rwth-i6/returnn/actions/runs/4406240591
def test_compare_eq():
    a_raw = torch.tensor([2.0, 2.0, 2.0])
    b_raw = torch.tensor([1.0, 2.0, 3.0])

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="float32")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="float32")

    result = a == b
    result_alt1 = rf.compare(a, "==", b)
    result_alt2 = rf.compare(a, "eq", b)

    assert result.raw_tensor.tolist() == [False, True, False]
    assert result_alt1.raw_tensor.tolist() == [False, True, False]
    assert result_alt2.raw_tensor.tolist() == [False, True, False]


def test_compare_ne():
    a_raw = torch.tensor([2.0, 2.0, 2.0])
    b_raw = torch.tensor([1.0, 2.0, 3.0])

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="float32")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="float32")

    result = a != b
    result_alt1 = rf.compare(a, "!=", b)
    result_alt2 = rf.compare(a, "<>", b)
    result_alt3 = rf.compare(a, "not_equal", b)

    assert result.raw_tensor.tolist() == [True, False, True]
    assert result_alt1.raw_tensor.tolist() == [True, False, True]
    assert result_alt2.raw_tensor.tolist() == [True, False, True]
    assert result_alt3.raw_tensor.tolist() == [True, False, True]


def test_compare_lt():
    a_raw = torch.tensor([2.0, 2.0, 2.0])
    b_raw = torch.tensor([1.0, 2.0, 3.0])

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="float32")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="float32")

    result = a < b
    result_alt1 = rf.compare(a, "<", b)
    result_alt2 = rf.compare(a, "less", b)

    assert result.raw_tensor.tolist() == [False, False, True]
    assert result_alt1.raw_tensor.tolist() == [False, False, True]
    assert result_alt2.raw_tensor.tolist() == [False, False, True]


def test_compare_le():
    a_raw = torch.tensor([2.0, 2.0, 2.0])
    b_raw = torch.tensor([1.0, 2.0, 3.0])

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="float32")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="float32")

    result = a <= b
    result_alt1 = rf.compare(a, "<=", b)
    result_alt2 = rf.compare(a, "less_equal", b)

    assert result.raw_tensor.tolist() == [False, True, True]
    assert result_alt1.raw_tensor.tolist() == [False, True, True]
    assert result_alt2.raw_tensor.tolist() == [False, True, True]


def test_compare_gt():
    a_raw = torch.tensor([2.0, 2.0, 2.0])
    b_raw = torch.tensor([1.0, 2.0, 3.0])

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="float32")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="float32")

    result = a > b
    result_alt1 = rf.compare(a, ">", b)
    result_alt2 = rf.compare(a, "greater", b)

    assert result.raw_tensor.tolist() == [True, False, False]
    assert result_alt1.raw_tensor.tolist() == [True, False, False]
    assert result_alt2.raw_tensor.tolist() == [True, False, False]


def test_compare_ge():
    a_raw = torch.tensor([2.0, 2.0, 2.0])
    b_raw = torch.tensor([1.0, 2.0, 3.0])

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="float32")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="float32")

    result = a >= b
    result_alt1 = rf.compare(a, ">=", b)
    result_alt2 = rf.compare(a, "greater_equal", b)

    assert result.raw_tensor.tolist() == [True, True, False]
    assert result_alt1.raw_tensor.tolist() == [True, True, False]
    assert result_alt2.raw_tensor.tolist() == [True, True, False]


def test_combine_add_int_tensors():
    a_raw = torch.tensor([2, 2, 2])
    b_raw = torch.tensor([1, 2, 3])

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="int64")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="int64")

    result = a + b
    result_alt1 = rf.combine(a, "+", b)
    result_alt2 = rf.combine(a, "add", b)

    assert result.raw_tensor.tolist() == pytest.approx([3, 4, 5])
    assert result_alt1.raw_tensor.tolist() == pytest.approx([3, 4, 5])
    assert result_alt2.raw_tensor.tolist() == pytest.approx([3, 4, 5])
    assert result.dtype == result_alt1.dtype == result_alt2.dtype == "int64"


def test_combine_add_float_tensors():
    a_raw = torch.tensor([2.0, 2.0, 2.0])
    b_raw = torch.tensor([1.0, 2.0, 3.0])

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="float32")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="float32")

    result = a + b
    result_alt1 = rf.combine(a, "+", b)
    result_alt2 = rf.combine(a, "add", b)

    assert result.raw_tensor.tolist() == pytest.approx([3.0, 4.0, 5.0])
    assert result_alt1.raw_tensor.tolist() == pytest.approx([3.0, 4.0, 5.0])
    assert result_alt2.raw_tensor.tolist() == pytest.approx([3.0, 4.0, 5.0])
    assert result.dtype == result_alt1.dtype == result_alt2.dtype == "float32"


def test_combine_add_number_to_tensor():
    a_raw = torch.tensor([2.0, 2.0, 2.0])
    b_raw = torch.tensor(3.0)

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="float32")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[], dtype="float32")

    result = a + b
    result_alt1 = rf.combine(a, "+", b)
    result_alt2 = rf.combine(a, "add", b)

    assert result.raw_tensor.tolist() == pytest.approx([5.0, 5.0, 5.0])
    assert result_alt1.raw_tensor.tolist() == pytest.approx([5.0, 5.0, 5.0])
    assert result_alt2.raw_tensor.tolist() == pytest.approx([5.0, 5.0, 5.0])
    assert result.dtype == result_alt1.dtype == result_alt2.dtype == "float32"


def test_combine_sub_int_tensors():
    a_raw = torch.tensor([2, 2, 2])
    b_raw = torch.tensor([1, 2, 3])

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="int64")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="int64")

    result = a - b
    result_alt1 = rf.combine(a, "-", b)
    result_alt2 = rf.combine(a, "sub", b)

    assert result.raw_tensor.tolist() == pytest.approx([1, 0, -1])
    assert result_alt1.raw_tensor.tolist() == pytest.approx([1, 0, -1])
    assert result_alt2.raw_tensor.tolist() == pytest.approx([1, 0, -1])
    assert result.dtype == result_alt1.dtype == result_alt2.dtype == "int64"


def test_combine_sub_float_tensors():
    a_raw = torch.tensor([2.0, 2.0, 2.0])
    b_raw = torch.tensor([1.0, 2.0, 3.0])

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="float32")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="float32")

    result = a - b
    result_alt1 = rf.combine(a, "-", b)
    result_alt2 = rf.combine(a, "sub", b)

    assert result.raw_tensor.tolist() == pytest.approx([1.0, 0.0, -1.0])
    assert result_alt1.raw_tensor.tolist() == pytest.approx([1.0, 0.0, -1.0])
    assert result_alt2.raw_tensor.tolist() == pytest.approx([1.0, 0.0, -1.0])
    assert result.dtype == result_alt1.dtype == result_alt2.dtype == "float32"


def test_combine_mul_int_tensors():
    a_raw = torch.tensor([2, 2, 2])
    b_raw = torch.tensor([1, 2, 3])

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="int64")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="int64")

    result = a * b
    result_alt1 = rf.combine(a, "*", b)
    result_alt2 = rf.combine(a, "mul", b)

    assert result.raw_tensor.tolist() == pytest.approx([2, 4, 6])
    assert result_alt1.raw_tensor.tolist() == pytest.approx([2, 4, 6])
    assert result_alt2.raw_tensor.tolist() == pytest.approx([2, 4, 6])
    assert result.dtype == result_alt1.dtype == result_alt2.dtype == "int64"


def test_combine_mul_float_tensors():
    a_raw = torch.tensor([2.0, 2.0, 2.0])
    b_raw = torch.tensor([1.0, 2.0, 3.0])

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="float32")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="float32")

    result = a * b
    result_alt1 = rf.combine(a, "*", b)
    result_alt2 = rf.combine(a, "mul", b)

    assert result.raw_tensor.tolist() == pytest.approx([2.0, 4.0, 6.0])
    assert result_alt1.raw_tensor.tolist() == pytest.approx([2.0, 4.0, 6.0])
    assert result_alt2.raw_tensor.tolist() == pytest.approx([2.0, 4.0, 6.0])
    assert result.dtype == result_alt1.dtype == result_alt2.dtype == "float32"


def test_combine_truediv_int_tensors():
    a_raw = torch.tensor([2, 2, 2])
    b_raw = torch.tensor([1, 2, 3])

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="int64")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="int64")

    # Until type promotion logic is implemented, we don't allow this.
    with pytest.raises(ValueError):
        a / b
    with pytest.raises(ValueError):
        rf.combine(a, "/", b)
    with pytest.raises(ValueError):
        rf.combine(a, "truediv", b)


def test_combine_truediv_float_tensors():
    a_raw = torch.tensor([2.0, 2.0, 2.0])
    b_raw = torch.tensor([1.0, 2.0, 3.0])

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="float32")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="float32")

    result = a / b
    result_alt1 = rf.combine(a, "/", b)
    result_alt2 = rf.combine(a, "truediv", b)

    assert result.raw_tensor.tolist() == pytest.approx([2.0, 1.0, 2.0 / 3.0])
    assert result_alt1.raw_tensor.tolist() == pytest.approx([2.0, 1.0, 2.0 / 3.0])
    assert result_alt2.raw_tensor.tolist() == pytest.approx([2.0, 1.0, 2.0 / 3.0])
    assert result.dtype == result_alt1.dtype == result_alt2.dtype == "float32"


def test_combine_floordiv_int_tensors():
    a_raw = torch.tensor([2, 2, 2])
    b_raw = torch.tensor([1, 2, 3])

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="int64")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="int64")

    result = a // b
    result_alt1 = rf.combine(a, "//", b)
    result_alt2 = rf.combine(a, "floordiv", b)

    assert result.raw_tensor.tolist() == pytest.approx([2, 1, 0])
    assert result_alt1.raw_tensor.tolist() == pytest.approx([2, 1, 0])
    assert result_alt2.raw_tensor.tolist() == pytest.approx([2, 1, 0])
    assert result.dtype == result_alt1.dtype == result_alt2.dtype == "int64"


def test_combine_floordiv_float_tensors():
    a_raw = torch.tensor([2.0, 2.0, 2.0])
    b_raw = torch.tensor([1.0, 2.0, 3.0])

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="float32")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="float32")

    result = a // b
    result_alt1 = rf.combine(a, "//", b)
    result_alt2 = rf.combine(a, "floordiv", b)

    assert result.raw_tensor.tolist() == pytest.approx([2.0, 1.0, 0.0])
    assert result_alt1.raw_tensor.tolist() == pytest.approx([2.0, 1.0, 0.0])
    assert result_alt2.raw_tensor.tolist() == pytest.approx([2.0, 1.0, 0.0])
    assert result.dtype == result_alt1.dtype == result_alt2.dtype == "float32"


def test_combine_mod_int_tensors():
    a_raw = torch.tensor([2, 2, 2, 17])
    b_raw = torch.tensor([1, 2, 3, 4])

    feature_dim = Dim(4)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="int64")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="int64")

    result = a % b
    result_alt1 = rf.combine(a, "%", b)
    result_alt2 = rf.combine(a, "mod", b)

    assert result.raw_tensor.tolist() == pytest.approx([0, 0, 2, 1])
    assert result_alt1.raw_tensor.tolist() == pytest.approx([0, 0, 2, 1])
    assert result_alt2.raw_tensor.tolist() == pytest.approx([0, 0, 2, 1])
    assert result.dtype == result_alt1.dtype == result_alt2.dtype == "int64"


def test_combine_mod_float_tensors():
    a_raw = torch.tensor([2.0, 2.0, 2.0, 17.0])
    b_raw = torch.tensor([1.0, 2.0, 3.0, 4.0])

    feature_dim = Dim(4)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="float32")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="float32")

    result = a % b
    result_alt1 = rf.combine(a, "%", b)
    result_alt2 = rf.combine(a, "mod", b)

    assert result.raw_tensor.tolist() == pytest.approx([0.0, 0.0, 2.0, 1.0])
    assert result_alt1.raw_tensor.tolist() == pytest.approx([0.0, 0.0, 2.0, 1.0])
    assert result_alt2.raw_tensor.tolist() == pytest.approx([0.0, 0.0, 2.0, 1.0])
    assert result.dtype == result_alt1.dtype == result_alt2.dtype == "float32"


def test_combine_pow_int_tensors():
    a_raw = torch.tensor([2, 2, 2])
    b_raw = torch.tensor([1, 2, 3])

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="int64")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="int64")

    result = a**b
    result_alt1 = rf.combine(a, "**", b)
    result_alt2 = rf.combine(a, "pow", b)

    assert result.raw_tensor.tolist() == pytest.approx([2, 4, 8])
    assert result_alt1.raw_tensor.tolist() == pytest.approx([2, 4, 8])
    assert result_alt2.raw_tensor.tolist() == pytest.approx([2, 4, 8])
    assert result.dtype == result_alt1.dtype == result_alt2.dtype == "int64"


def test_combine_pow_float_tensors():
    a_raw = torch.tensor([2.0, 2.0, 2.0])
    b_raw = torch.tensor([1.0, 2.0, 3.0])

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="float32")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="float32")

    result = a**b
    result_alt1 = rf.combine(a, "**", b)
    result_alt2 = rf.combine(a, "pow", b)

    assert result.raw_tensor.tolist() == pytest.approx([2.0, 4.0, 8.0])
    assert result_alt1.raw_tensor.tolist() == pytest.approx([2.0, 4.0, 8.0])
    assert result_alt2.raw_tensor.tolist() == pytest.approx([2.0, 4.0, 8.0])
    assert result.dtype == result_alt1.dtype == result_alt2.dtype == "float32"


def test_combine_max():
    a_raw = torch.tensor([2.0, 2.0, 2.0])
    b_raw = torch.tensor([1.0, 2.0, 3.0])

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="float32")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="float32")

    result = rf.combine(a, "max", b)
    result_alt1 = rf.combine(a, "maximum", b)

    assert result.raw_tensor.tolist() == [2.0, 2.0, 3.0]
    assert result_alt1.raw_tensor.tolist() == [2.0, 2.0, 3.0]


def test_combine_min():
    a_raw = torch.tensor([2.0, 2.0, 2.0])
    b_raw = torch.tensor([1.0, 2.0, 3.0])

    feature_dim = Dim(3)

    a = Tensor(name="a", raw_tensor=a_raw, dims=[feature_dim], dtype="float32")
    b = Tensor(name="b", raw_tensor=b_raw, dims=[feature_dim], dtype="float32")

    result = rf.combine(a, "min", b)
    result_alt1 = rf.combine(a, "minimum", b)

    assert result.raw_tensor.tolist() == [1.0, 2.0, 2.0]
    assert result_alt1.raw_tensor.tolist() == [1.0, 2.0, 2.0]


def test_native_is_raw_torch_tensor_type():
    raw_tensor = torch.zeros(2, 3)
    raw_parameter = torch.nn.Parameter(torch.zeros(2, 3))
    numpy_tensor = numpy.zeros((2, 3))

    from returnn.frontend import _native

    mod = _native.get_module(verbose=True)
    # Check that there is no Python code executed via sys.settrace.
    trace_num_calls = 0

    def tracefunc(frame, event, arg):
        print("*** trace:", frame, event, arg)
        nonlocal trace_num_calls
        trace_num_calls += 1

    old_tracefunc = sys.gettrace()
    try:
        sys.settrace(tracefunc)
        assert mod.is_raw_torch_tensor_type(type(raw_tensor)) is True
        assert mod.is_raw_torch_tensor_type(type(raw_parameter)) is True
        assert mod.is_raw_torch_tensor_type(type(numpy_tensor)) is False
        assert mod.is_raw_torch_tensor_type(type(43)) is False
        assert mod.is_raw_torch_tensor_type(43) is False  # current behavior - might also raise exception instead
    finally:
        sys.settrace(old_tracefunc)
    assert trace_num_calls == 0


def test_native_torch_raw_backend():
    tensor = Tensor(name="a", raw_tensor=torch.tensor([1.0, 2.0, 3.0]), dims=[Dim(3)], dtype="float32")
    backend1 = tensor._raw_backend

    import returnn.frontend._backend as _backend_api

    backend2 = _backend_api.get_backend_by_raw_tensor_type(type(tensor.raw_tensor))

    from returnn.frontend import _native

    mod = _native.get_module(verbose=True)
    # Check that there is no Python code executed via sys.settrace.
    trace_num_calls = 0

    def tracefunc(frame, event, arg):
        print("*** trace:", frame, event, arg)
        nonlocal trace_num_calls
        trace_num_calls += 1

    old_tracefunc = sys.gettrace()
    try:
        sys.settrace(tracefunc)
        backend3 = mod.get_backend_for_tensor(tensor)
    finally:
        sys.settrace(old_tracefunc)

    assert backend1 is backend2 is backend3
    assert trace_num_calls == 0


def test_native_torch_tensor_eq():
    batch_dim = Dim(2, name="batch_dim")
    feature_dim = Dim(3, name="feature_dim")
    tensor_bf = Tensor("tensor", dims=[batch_dim, feature_dim], dtype="float32", raw_tensor=torch.zeros(2, 3))
    tensor_f = Tensor(
        "tensor", dims=[feature_dim], dtype="float32", raw_tensor=torch.arange(-1, 2, dtype=torch.float32)
    )

    from returnn.frontend import _native

    mod = _native.get_module(verbose=True)
    # Check that there is no Python code executed via sys.settrace.
    trace_num_calls = 0

    def tracefunc(frame, event, arg):
        print("*** trace:", frame, event, arg)
        nonlocal trace_num_calls
        trace_num_calls += 1

    old_tracefunc = sys.gettrace()
    try:
        sys.settrace(tracefunc)
        mod.tensor_eq(tensor_bf, tensor_bf)
        mod.tensor_eq(tensor_bf, tensor_f)
        mod.tensor_eq(tensor_bf, 0.0)
    finally:
        sys.settrace(old_tracefunc)
    assert trace_num_calls == 0


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
