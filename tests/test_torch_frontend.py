"""
tests for returnn.torch.frontend
"""

import _setup_test_env  # noqa

import torch
import pytest
import math

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


def test_dot_scalar_multiplication():
    a_raw = torch.tensor(2.0)
    b_raw = torch.tensor(3.0)

    a = Tensor(name="a", dims=[], dtype="float32", raw_tensor=a_raw)
    b = Tensor(name="b", dims=[], dtype="float32", raw_tensor=b_raw)

    result = rf.matmul(a, b, reduce=[])

    assert pytest.approx(result.raw_tensor) == 6.0


def test_dot_scalar_product():
    a_raw = torch.tensor([1.0, 2.0, 3.0])
    b_raw = torch.tensor([4.0, 5.0, 6.0])

    feature_dim = Dim(dimension=3)

    a = Tensor(name="a", dims=[feature_dim], dtype="float32", raw_tensor=a_raw)
    b = Tensor(name="b", dims=[feature_dim], dtype="float32", raw_tensor=b_raw)

    result = rf.matmul(a, b, reduce=[feature_dim])

    assert pytest.approx(result.raw_tensor) == 32.0


def test_dot_outer_product():
    a_raw = torch.tensor([1.0, 2.0, 3.0])
    b_raw = torch.tensor([4.0, 5.0, 6.0])

    a_feature_dim = Dim(dimension=3)
    b_feature_dim = Dim(dimension=3)

    a = Tensor(name="a", dims=[a_feature_dim], dtype="float32", raw_tensor=a_raw)
    b = Tensor(name="b", dims=[b_feature_dim], dtype="float32", raw_tensor=b_raw)

    result = rf.matmul(a, b, reduce=[])

    assert result.dims == (a_feature_dim, b_feature_dim)
    assert result.raw_tensor.shape == (3, 3)


def test_dot_matrix_vector_product():
    a_raw = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    b_raw = torch.tensor([4.0, 5.0])

    a_feature_dim = Dim(dimension=3)
    reduce_dim = Dim(dimension=2)

    a = Tensor(name="a", dims=[reduce_dim, a_feature_dim], dtype="float32", raw_tensor=a_raw)
    b = Tensor(name="b", dims=[reduce_dim], dtype="float32", raw_tensor=b_raw)

    result = rf.matmul(a, b, reduce=[reduce_dim])

    assert result.dims == (a_feature_dim,)
    assert result.raw_tensor.tolist() == pytest.approx([-1.0, -2.0, -3.0])


def test_dot_matrix_matrix_product():
    a_raw = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    b_raw = torch.tensor([[1.0, -2.0], [2.0, -4.0]])

    a_feature_dim = Dim(dimension=3)
    b_feature_dim = Dim(dimension=2)
    reduce_dim = Dim(dimension=2)

    a = Tensor(name="a", dims=[a_feature_dim, reduce_dim], dtype="float32", raw_tensor=a_raw)
    b = Tensor(name="b", dims=[reduce_dim, b_feature_dim], dtype="float32", raw_tensor=b_raw)

    result = rf.matmul(a, b, reduce=[reduce_dim])

    assert result.dims == (a_feature_dim, b_feature_dim)
    assert torch.allclose(result.raw_tensor, torch.tensor([[5.0, -10.0], [11.0, -22.0], [17.0, -34.0]]))


def test_dot_scale_matrix():
    a_raw = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    b_raw = torch.tensor(2.0)

    a_feature_dim1 = Dim(dimension=2)
    a_feature_dim2 = Dim(dimension=3)

    a = Tensor(name="a", dims=[a_feature_dim1, a_feature_dim2], dtype="float32", raw_tensor=a_raw)
    b = Tensor(name="b", dims=[], dtype="float32", raw_tensor=b_raw)

    result = rf.matmul(a, b, reduce=[])

    assert result.dims == (a_feature_dim1, a_feature_dim2)
    assert torch.allclose(result.raw_tensor, torch.tensor([[2.0, 4.0, 6.0], [-2.0, -4.0, -6.0]]))


def test_dot_batched_scalar_multiplication():
    a_raw = torch.tensor([1.0, 2.0, 3.0])
    b_raw = torch.tensor([4.0, 5.0, 6.0])

    batch_dim = Dim(dimension=3)

    a = Tensor(name="a", dims=[batch_dim], dtype="float32", raw_tensor=a_raw)
    b = Tensor(name="b", dims=[batch_dim], dtype="float32", raw_tensor=b_raw)

    result = rf.matmul(a, b, reduce=[])

    assert result.dims == (batch_dim,)
    assert result.raw_tensor.tolist() == pytest.approx([4.0, 10.0, 18.0])


def test_dot_batched_scalar_product():
    a_raw = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    b_raw = torch.tensor([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])

    batch_dim = Dim(dimension=2)
    feature_dim = Dim(dimension=3)

    a = Tensor(name="a", dims=[batch_dim, feature_dim], dtype="float32", raw_tensor=a_raw)
    b = Tensor(name="b", dims=[batch_dim, feature_dim], dtype="float32", raw_tensor=b_raw)

    result = rf.matmul(a, b, reduce=[feature_dim])

    assert result.dims == (batch_dim,)
    assert result.raw_tensor.tolist() == pytest.approx([32.0, -32.0])


def test_dot_batched_outer_product():
    a_raw = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    b_raw = torch.tensor([[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]])

    batch_dim = Dim(dimension=2)
    a_feature_dim = Dim(dimension=3)
    b_feature_dim = Dim(dimension=3)

    a = Tensor(name="a", dims=[batch_dim, a_feature_dim], dtype="float32", raw_tensor=a_raw)
    b = Tensor(name="b", dims=[batch_dim, b_feature_dim], dtype="float32", raw_tensor=b_raw)

    result = rf.matmul(a, b, reduce=[])

    assert result.dims == (batch_dim, a_feature_dim, b_feature_dim)
    assert result.raw_tensor.shape == (2, 3, 3)


def test_dot_batched_matrix_vector_product():
    a_raw = torch.tensor([[[1.0, -1.0], [2.0, -2.0]], [[3.0, -3.0], [4.0, -4.0]], [[5.0, -5.0], [6.0, -6.0]]])
    b_raw = torch.tensor([[1.0, 2.0], [2.0, 4.0]])

    batch_dim = Dim(dimension=2)
    a_feature_dim = Dim(dimension=3)
    reduce_dim = Dim(dimension=2)

    # intentionally test strange batch axis
    a = Tensor(name="a", dims=[a_feature_dim, reduce_dim, batch_dim], dtype="float32", raw_tensor=a_raw)
    b = Tensor(name="b", dims=[batch_dim, reduce_dim], dtype="float32", raw_tensor=b_raw)

    result = rf.matmul(a, b, reduce=[reduce_dim])

    assert result.dims == (batch_dim, a_feature_dim)
    assert torch.allclose(result.raw_tensor, torch.tensor([[5.0, 11.0, 17.0], [-10.0, -22.0, -34.0]]))


def test_dot_batched_matrix_matrix_product():
    a_raw = torch.tensor([[[1.0, 2.0], [-1.0, -2.0]], [[3.0, 4.0], [-3.0, -4.0]], [[5.0, 6.0], [-5.0, -6.0]]])
    b_raw = torch.tensor([[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]], [[5.0, 5.0], [6.0, 6.0]]])

    batch_dim = Dim(dimension=2)
    a_feature_dim = Dim(dimension=3)
    b_feature_dim = Dim(dimension=3)
    reduce_dim = Dim(dimension=2)

    # intentionally test strange batch axis
    a = Tensor(name="a", dims=[a_feature_dim, reduce_dim, batch_dim], dtype="float32", raw_tensor=a_raw)
    b = Tensor(name="b", dims=[b_feature_dim, batch_dim, reduce_dim], dtype="float32", raw_tensor=b_raw)

    result = rf.matmul(a, b, reduce=[reduce_dim])

    assert result.dims == (batch_dim, a_feature_dim, b_feature_dim)
    assert torch.allclose(result.raw_tensor, torch.zeros(size=(2, 3, 3)))  # values chosen such that everything cancels


def test_dot_batched_scale_matrix():
    a_raw = torch.tensor([2.0, 3.0])
    b_raw = torch.tensor([[[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], [[2.0, 3.0, 4.0], [-2.0, -3.0, -4.0]]])

    batch_dim = Dim(dimension=2)
    b_feature_dim1 = Dim(dimension=2)
    b_feature_dim2 = Dim(dimension=3)

    a = Tensor(name="a", dims=[batch_dim], dtype="float32", raw_tensor=a_raw)
    b = Tensor(name="b", dims=[batch_dim, b_feature_dim1, b_feature_dim2], dtype="float32", raw_tensor=b_raw)

    result = rf.matmul(a, b, reduce=[])

    assert result.dims == (batch_dim, b_feature_dim1, b_feature_dim2)
    assert torch.allclose(
        result.raw_tensor,
        torch.tensor([[[2.0, 4.0, 6.0], [-2.0, -4.0, -6.0]], [[6.0, 9.0, 12.0], [-6.0, -9.0, -12.0]]]),
    )


def test_dot_multiple_dims():
    a_raw = torch.rand(size=(2, 4, 6, 9, 5, 3, 8, 1))
    b_raw = torch.rand(size=(7, 2, 6, 8, 3, 1, 5, 4))

    reduce_dim_1 = Dim(dimension=3)
    reduce_dim_2 = Dim(dimension=6)
    reduce_dim_3 = Dim(dimension=1)

    common_dim_1 = Dim(dimension=2)
    common_dim_2 = Dim(dimension=8)
    common_dim_3 = Dim(dimension=5)

    a_unique_dim_1 = Dim(dimension=9)
    a_unique_dim_2 = Dim(dimension=4)

    b_unique_dim_1 = Dim(dimension=7)
    b_unique_dim_2 = Dim(dimension=4)

    a = Tensor(
        name="a",
        dims=[
            common_dim_1,
            a_unique_dim_2,
            reduce_dim_2,
            a_unique_dim_1,
            common_dim_3,
            reduce_dim_1,
            common_dim_2,
            reduce_dim_3,
        ],
        dtype="float32",
        raw_tensor=a_raw,
    )
    b = Tensor(
        name="b",
        dims=[
            b_unique_dim_1,
            common_dim_1,
            reduce_dim_2,
            common_dim_2,
            reduce_dim_1,
            reduce_dim_3,
            common_dim_3,
            b_unique_dim_2,
        ],
        dtype="float32",
        raw_tensor=b_raw,
    )

    result = rf.matmul(a, b, reduce=[reduce_dim_1, reduce_dim_2, reduce_dim_3])

    # assumes common dims as sorted in a, unique dims as sorted in a / b respectively
    assert result.dims == (
        common_dim_1,
        common_dim_3,
        common_dim_2,
        a_unique_dim_2,
        a_unique_dim_1,
        b_unique_dim_1,
        b_unique_dim_2,
    )
    assert result.raw_tensor.shape == (2, 5, 8, 4, 9, 7, 4)


def test_cross_entropy_no_batch_dim():
    logits_raw = torch.tensor([0.0, 0.0, math.log(10.0), 0.0, 0.0, 0.0])
    target_raw = torch.tensor(2, dtype=torch.int64)

    classes_dim = Dim(dimension=6)

    logits = Tensor(name="logits", dims=[classes_dim], dtype="float32", raw_tensor=logits_raw)
    target = Tensor(name="target", dims=[], sparse_dim=classes_dim, dtype="int64", raw_tensor=target_raw)

    cross_entropy = rf.cross_entropy(estimated=logits, target=target, axis=classes_dim, estimated_type="logits")

    assert not cross_entropy.dims
    assert cross_entropy.raw_tensor.tolist() == pytest.approx(-math.log(10 / 15))


def test_cross_entropy_no_batch_dim_dense_target():
    logits_raw = torch.tensor([0.0, 0.0, math.log(10.0), 0.0, 0.0, 0.0])
    target_raw = torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.5])

    classes_dim = Dim(dimension=6)

    logits = Tensor(name="logits", dims=[classes_dim], dtype="float32", raw_tensor=logits_raw)
    target = Tensor(name="target", dims=[classes_dim], dtype="float32", raw_tensor=target_raw)

    cross_entropy = rf.cross_entropy(estimated=logits, target=target, axis=classes_dim, estimated_type="logits")

    assert not cross_entropy.dims
    assert cross_entropy.raw_tensor.tolist() == pytest.approx(-0.5 * math.log(10 / 15) - 0.5 * math.log(1 / 15))


def test_cross_entropy():
    logits_raw = torch.tensor([[0.0, 0.0, math.log(3.0)], [math.log(5.0), 0.0, 0.0], [0.0, math.log(2.0), 0.0]])
    target_raw = torch.tensor([2, 0, 1], dtype=torch.int64)

    batch_dim = Dim(dimension=3)
    classes_dim = Dim(dimension=3)

    logits = Tensor(name="logits", dims=[batch_dim, classes_dim], dtype="float32", raw_tensor=logits_raw)
    target = Tensor(name="target", dims=[batch_dim], sparse_dim=classes_dim, dtype="int64", raw_tensor=target_raw)

    cross_entropy = rf.cross_entropy(estimated=logits, target=target, axis=classes_dim, estimated_type="logits")

    assert cross_entropy.dims == (batch_dim,)
    assert cross_entropy.raw_tensor.tolist() == pytest.approx([-math.log(3 / 5), -math.log(5 / 7), -math.log(2 / 4)])


def test_cross_entropy_dense_target():
    logits_raw = torch.tensor([[0.0, math.log(5.0)], [0.0, 0.0], [math.log(3.0), 0.0]])
    target_raw = torch.tensor([[0.0, 0.4, 0.6], [0.3, 0.7, 0.0]])

    batch_dim = Dim(dimension=2)
    classes_dim = Dim(dimension=3)

    logits = Tensor(name="logits", dims=[classes_dim, batch_dim], dtype="float32", raw_tensor=logits_raw)
    target = Tensor(name="target", dims=[batch_dim, classes_dim], dtype="float32", raw_tensor=target_raw)

    cross_entropy = rf.cross_entropy(estimated=logits, target=target, axis=classes_dim, estimated_type="logits")

    assert cross_entropy.dims == (batch_dim,)
    cross_entropy_list = cross_entropy.raw_tensor.tolist()
    assert cross_entropy_list[0] == pytest.approx(-0.6 * math.log(3 / 5) - 0.4 * math.log(1 / 5))
    assert cross_entropy_list[1] == pytest.approx(-0.3 * math.log(5 / 7) - 0.7 * math.log(1 / 7))
