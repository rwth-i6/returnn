"""
tests for returnn.torch.frontend
"""

import _setup_test_env  # noqa

import torch
import pytest

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


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
