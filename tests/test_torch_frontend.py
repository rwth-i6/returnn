"""
tests for returnn.torch.frontend
"""

import _setup_test_env  # noqa

from typing import Any, Dict, List
import numpy.testing
import torch
import pytest
import math
import sys
import unittest

from torch_utils import (
    report_profile,
    get_remaining_allocs_from_profile,
    get_allocs_from_profile,
    get_peak_alloc_from_profile,
)

from returnn.util import better_exchook
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


_torch_default_device = torch.device("cpu")


def _setup():
    rf.select_backend_torch()
    global _torch_default_device
    if torch.cuda.is_available():
        _torch_default_device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        _torch_default_device = torch.device("mps")


_setup()


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


def test_pack_padded_wrong_grad():
    # https://github.com/pytorch/pytorch/issues/99638

    # noinspection PyShadowingNames
    def _loss_rf_packed(logits: Tensor, targets: Tensor) -> torch.Tensor:
        logits_packed, pack_dim = rf.pack_padded(logits, dims=(batch_dim, time_dim), enforce_sorted=False)
        targets_packed, _ = rf.pack_padded(targets, dims=(batch_dim, time_dim), enforce_sorted=False, out_dim=pack_dim)
        loss_rf_packed = rf.cross_entropy(
            estimated=logits_packed, estimated_type="logits", target=targets_packed, axis=classes_dim
        )
        loss_rf_packed_sum = rf.reduce_sum(loss_rf_packed, axis=loss_rf_packed.dims)
        return loss_rf_packed_sum.raw_tensor

    # noinspection PyShadowingNames
    def _loss_pt_packed(logits: Tensor, targets: Tensor) -> torch.Tensor:
        logits_pt_packed_raw = torch.nn.utils.rnn.pack_padded_sequence(
            logits.raw_tensor, time_dim.dyn_size, batch_first=True, enforce_sorted=False
        )
        targets_pt_packed_raw = torch.nn.utils.rnn.pack_padded_sequence(
            targets.raw_tensor, time_dim.dyn_size, batch_first=True, enforce_sorted=False
        )
        loss_pt_packed_raw = torch.nn.CrossEntropyLoss(reduction="none")(
            logits_pt_packed_raw.data, targets_pt_packed_raw.data.long()
        )
        loss_pt_packed_sum_raw = torch.sum(loss_pt_packed_raw)
        return loss_pt_packed_sum_raw

    # noinspection PyShadowingNames
    def _loss_rf_padded(logits: Tensor, targets: Tensor) -> torch.Tensor:
        loss_rf_padded = rf.cross_entropy(estimated=logits, estimated_type="logits", target=targets, axis=classes_dim)
        loss_rf_padded_sum = rf.reduce_sum(loss_rf_padded, axis=loss_rf_padded.dims)
        return loss_rf_padded_sum.raw_tensor

    prev_loss_value = None
    prev_bias_grad = None

    for loss_fn in [_loss_pt_packed, _loss_rf_padded, _loss_rf_packed]:
        torch.manual_seed(42)

        batch_dim = Dim(dimension=3, name="batch")
        in_dim = Dim(dimension=5, name="in")
        classes_dim = Dim(dimension=5, name="classes")

        net = torch.nn.Conv1d(in_dim.dimension, classes_dim.dimension, 5, padding="same")

        time_dim = Dim(
            Tensor(name="time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([4, 3, 2], dtype=torch.int32))
        )
        inputs = Tensor(
            name="inputs",
            dims=[batch_dim, time_dim, classes_dim],
            dtype="float32",
            raw_tensor=torch.randn(3, 4, 5, requires_grad=True),
        )
        targets = Tensor(
            name="target",
            dims=[batch_dim, time_dim],
            sparse_dim=classes_dim,
            dtype="int64",
            raw_tensor=torch.randint(0, 5, (3, 4)),
        )

        logits_raw_ = net(inputs.raw_tensor.transpose(1, 2))
        logits_raw = logits_raw_.transpose(1, 2)
        logits = Tensor(
            name="logits",
            dims=[batch_dim, time_dim, classes_dim],
            dtype="float32",
            raw_tensor=logits_raw,
        )

        loss_raw = loss_fn(logits, targets)
        loss_value = loss_raw.detach().cpu().numpy()
        print("loss:", loss_raw)

        (bias_grad,) = torch.autograd.grad(loss_raw, net.bias, create_graph=True)
        print("bias grad:", bias_grad)
        bias_grad = bias_grad.detach().cpu().numpy()

        if prev_loss_value is not None:
            numpy.testing.assert_almost_equal(loss_value, prev_loss_value, decimal=5, err_msg="loss")
            numpy.testing.assert_almost_equal(bias_grad, prev_bias_grad, decimal=5, err_msg="bias grad")

        prev_loss_value = loss_value
        prev_bias_grad = bias_grad


@unittest.skipIf(torch.__version__ < (2,), "report_profile needs PyTorch >= 2.0")
def test_pack_padded_memory():
    import numpy as np
    import torch
    from returnn.tensor import Dim

    rnd = np.random.RandomState(42)
    batch_dim_ = Dim(5, name="batch")
    batch_dims = [batch_dim_]
    vocab_dim = Dim(7, name="vocab")
    enc_dim = Dim(rf.convert_to_tensor(torch.tensor([17, 16, 15, 13, 12], device="cpu"), dims=[batch_dim_]), name="enc")
    dec_dim = Dim(rf.convert_to_tensor(torch.tensor([11, 10, 8, 7, 5], device="cpu"), dims=[batch_dim_]), name="dec")
    logits = rf.convert_to_tensor(
        torch.tensor(
            rnd.randn(
                batch_dim_.dimension,
                enc_dim.dyn_size_ext.raw_tensor.max(),
                dec_dim.dyn_size_ext.raw_tensor.max(),
                vocab_dim.dimension,
            ).astype(np.float32),
            device=_torch_default_device,
        ),
        dims=[batch_dim_, enc_dim, dec_dim, vocab_dim],
    )
    print("dev:", logits.device)
    sizeof_float = 4

    def _get_rf_pack_packed() -> torch.Tensor:
        logits_packed, pack_dim = rf.pack_padded(
            logits, dims=batch_dims + [enc_dim, dec_dim], enforce_sorted=False
        )  # [B * T * (S+1), D]
        return logits_packed.raw_tensor

    def _get_naive_pack_padded() -> torch.Tensor:
        logits_raw = logits.copy_transpose(batch_dims + [enc_dim, dec_dim, vocab_dim]).raw_tensor
        enc_lens = enc_dim.dyn_size_ext.raw_tensor
        non_blank_lens = dec_dim.dyn_size_ext.raw_tensor
        vocab_len = vocab_dim.dimension

        batch_tensors = []

        for b in range(logits_raw.shape[0]):
            enc_len = enc_lens[b]
            non_blank_len = non_blank_lens[b]
            combined_len = enc_len * non_blank_len
            logits_single = logits_raw[b, :enc_len, :non_blank_len]
            logits_single = torch.reshape(logits_single, (combined_len, vocab_len))
            batch_tensors.append(logits_single)

        return torch.cat(batch_tensors, dim=0)

    from torch.profiler import profile, ProfilerActivity

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
    ) as prof_rf:
        rf_pack_padded_res = _get_rf_pack_packed()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
    ) as prof_naive:
        naive_pack_padded_res = _get_naive_pack_padded()

    print("result shape:", rf_pack_padded_res.shape, "numel:", rf_pack_padded_res.numel())
    assert rf_pack_padded_res.shape == naive_pack_padded_res.shape
    assert rf_pack_padded_res.device == naive_pack_padded_res.device
    assert torch.eq(rf_pack_padded_res, naive_pack_padded_res).all()

    print("*** RF ***")
    report_profile(prof_rf, allow_remaining_allocs=True)
    print("*** Naive ***")
    report_profile(prof_naive, allow_remaining_allocs=True)
    print("***")

    def _filter_rf_alloc(alloc: Dict[str, Any]) -> bool:
        # Filter some RF internal caches which will get created.
        return "/sequence_mask/get_mask/" not in alloc["name"]

    def _filter_rf_allocs_dict(allocs: Dict[int, Dict[str, Any]]):
        return {k: v for k, v in allocs.items() if _filter_rf_alloc(v)}

    def _filter_rf_allocs_list(allocs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [v for v in allocs if _filter_rf_alloc(v)]

    print("inputs shape:", logits.raw_tensor.shape, "numel:", logits.raw_tensor.numel())
    print("  byte size:", logits.raw_tensor.numel() * sizeof_float)
    print("result shape:", rf_pack_padded_res.shape, "numel:", rf_pack_padded_res.numel())
    print("  byte size:", rf_pack_padded_res.numel() * sizeof_float)
    enc_lens = enc_dim.dyn_size_ext.raw_tensor
    non_blank_lens = dec_dim.dyn_size_ext.raw_tensor
    print("Mask size:", batch_dim_.dimension * max(enc_lens) * max(non_blank_lens))
    total_num_el = 0
    for b in range(batch_dim_.dimension):
        enc_len = enc_lens[b]
        non_blank_len = non_blank_lens[b]
        total_num_el += enc_len * non_blank_len
    print("Expected total num elements:", total_num_el, "with vocab:", total_num_el * vocab_dim.dimension)
    print("Size:", total_num_el * vocab_dim.dimension * sizeof_float)

    print("Remaining allocs:")
    allocs_rf = get_remaining_allocs_from_profile(prof_rf)
    allocs_rf = _filter_rf_allocs_dict(allocs_rf)
    print("RF:", allocs_rf)
    allocs_naive = get_remaining_allocs_from_profile(prof_naive)
    print("Naive:", allocs_naive)
    assert len(allocs_rf) == len(allocs_naive) == 1
    assert (
        list(allocs_rf.values())[0]["size"] == list(allocs_naive.values())[0]["size"]
        # On CPU, it should match, but on GPU, it will allocate more.
        # == rf_pack_padded_res.numel() * sizeof_float
    )

    print("All allocs RF:")
    for alloc in _filter_rf_allocs_list(get_allocs_from_profile(prof_rf)):
        alloc["name"] = alloc["name"][alloc["name"].find("/_get_rf_pack_packed/") + len("/_get_rf_pack_packed/") :]
        print(" ", alloc)
    print("All allocs naive:")
    for alloc in get_allocs_from_profile(prof_naive):
        alloc["name"] = alloc["name"][
            alloc["name"].find("/_get_naive_pack_padded/") + len("/_get_naive_pack_padded/") :
        ]
        print(" ", alloc)

    print("Peak alloc:")
    print("RF:", get_peak_alloc_from_profile(prof_rf))
    print("Naive:", get_peak_alloc_from_profile(prof_naive))

    print("dev:", rf_pack_padded_res.device)


def _benchmark_pack_padded():
    from torch.utils.benchmark import Timer
    import numpy as np
    import torch
    from returnn.tensor import Dim

    rnd = np.random.RandomState(42)
    batch_dim_ = Dim(113, name="batch")
    batch_dims = [batch_dim_]
    vocab_dim = Dim(1023, name="vocab")
    enc_dim = Dim(
        rf.convert_to_tensor(
            torch.tensor(rnd.randint(11, 55, size=[batch_dim_.dimension]), device="cpu"), dims=[batch_dim_]
        ),
        name="enc",
    )
    dec_dim = Dim(
        rf.convert_to_tensor(
            torch.tensor(rnd.randint(7, 23, size=[batch_dim_.dimension]), device="cpu"), dims=[batch_dim_]
        ),
        name="dec",
    )
    logits = rf.convert_to_tensor(
        torch.tensor(
            rnd.randn(
                batch_dim_.dimension,
                enc_dim.dyn_size_ext.raw_tensor.max(),
                dec_dim.dyn_size_ext.raw_tensor.max(),
                vocab_dim.dimension,
            ).astype(np.float32),
            device=_torch_default_device,
        ),
        dims=[batch_dim_, enc_dim, dec_dim, vocab_dim],
    )
    eye = rf.convert_to_tensor(
        torch.eye(vocab_dim.dimension, device=logits.raw_tensor.device, dtype=logits.raw_tensor.dtype),
        dims=[vocab_dim.copy(match_priority=1), vocab_dim],
    )
    sizeof_float = 4
    print("logits shape:", logits.raw_tensor.shape)
    print("logits size:", logits.raw_tensor.numel() * sizeof_float, "bytes")
    print("dev:", logits.device)

    # Call this once because this will cache some things.
    # Exclude this part for the benchmark to have it fair.
    dims = batch_dims + [enc_dim, dec_dim]
    rf.sequence_mask(dims, device=logits.device)

    def _no_op_test():
        pass

    def _get_logits() -> Tensor:
        # Maybe do sth with logits, to better see effects of CUDA host-device synchronization.
        # return rf.matmul(logits, eye, reduce=vocab_dim)
        # return logits * 0.9 + 0.1
        return logits

    def _get_rf_pack_packed() -> torch.Tensor:
        logits_ = _get_logits()
        logits_packed, pack_dim = rf.pack_padded(logits_, dims=dims, enforce_sorted=False)  # [B * T * S, D]
        return logits_packed.raw_tensor

    def _get_rf_pack_padded_known_lens() -> torch.Tensor:
        logits_ = _get_logits()
        mask = rf.sequence_mask(dims, device=logits.device)
        assert mask.dims_set == set(dims)
        # Note: Already calculating out_dim here can trigger a more efficient calculation path in masked_select,
        # where we can avoid a CUDA host-device synchronization, e.g. in the PyTorch backend.
        # See https://github.com/rwth-i6/returnn/pull/1593.
        pack_dim = Dim(rf.num_elements_of_shape(dims), name="packed")
        logits_packed, _ = rf.masked_select(logits_, mask=mask, dims=dims, out_dim=pack_dim)
        return logits_packed.raw_tensor

    def _get_rf_pack_padded_no_known_lens() -> torch.Tensor:
        logits_ = _get_logits()
        mask = rf.sequence_mask(dims, device=logits.device)
        assert mask.dims_set == set(dims)
        logits_packed, pack_dim = rf.masked_select(logits_, mask=mask, dims=dims)
        return logits_packed.raw_tensor

    def _get_torch_masked_select_pack_padded() -> torch.Tensor:
        logits_ = _get_logits()
        # This was the old implementation of rf.pack_padded before https://github.com/rwth-i6/returnn/pull/1586.
        remaining_dims = [vocab_dim]
        tensor_templ_dims = dims + remaining_dims
        mask = rf.sequence_mask(dims, device=logits.device)
        in_raw = logits_.copy_compatible_to_dims_raw(tensor_templ_dims)
        mask_raw = mask.copy_compatible_to_dims_raw(tensor_templ_dims)
        out_raw = torch.masked_select(in_raw, mask_raw)
        remaining_shape = [d.get_dim_value() for d in remaining_dims]
        remaining_num_elements = numpy.prod(remaining_shape) if remaining_shape else 1
        assert out_raw.numel() % remaining_num_elements == 0
        flattened_num_elements = out_raw.numel() // remaining_num_elements
        out_raw = torch.reshape(out_raw, [flattened_num_elements] + remaining_shape)
        return out_raw

    def _get_naive_pack_padded() -> torch.Tensor:
        logits_ = _get_logits()
        tensor_templ_dims = dims + [vocab_dim]
        logits_raw = logits_.copy_compatible_to_dims_raw(tensor_templ_dims)
        enc_lens = enc_dim.dyn_size_ext.raw_tensor
        non_blank_lens = dec_dim.dyn_size_ext.raw_tensor
        vocab_len = vocab_dim.dimension

        batch_tensors = []

        for b in range(logits_raw.shape[0]):
            enc_len = enc_lens[b]
            non_blank_len = non_blank_lens[b]
            combined_len = enc_len * non_blank_len
            logits_single = logits_raw[b, :enc_len, :non_blank_len]
            logits_single = torch.reshape(logits_single, (combined_len, vocab_len))
            batch_tensors.append(logits_single)

        return torch.cat(batch_tensors, dim=0)

    for f in [
        _no_op_test,  # test
        _get_logits,  # warmup dummy
        _get_rf_pack_packed,
        _get_rf_pack_padded_known_lens,
        _get_rf_pack_padded_no_known_lens,
        _get_torch_masked_select_pack_padded,
        _get_naive_pack_padded,
    ]:
        print("func:", f.__name__)
        t = Timer(stmt="func()", globals={"func": f})
        try:
            print(t.blocked_autorange(min_run_time=0.5))
        except torch.cuda.OutOfMemoryError:
            print("CUDA OOM")


def _benchmark_pack_padded_one_dim():
    from torch.utils.benchmark import Timer
    import numpy as np
    import torch
    from returnn.tensor import Dim

    rnd = np.random.RandomState(42)
    batch_dim_ = Dim(113, name="batch")
    batch_dims = [batch_dim_]
    vocab_dim = Dim(1023, name="vocab")
    enc_dim = Dim(
        rf.convert_to_tensor(
            torch.tensor(rnd.randint(11, 1011, size=[batch_dim_.dimension]), device="cpu"), dims=[batch_dim_]
        ),
        name="enc",
    )
    logits = rf.convert_to_tensor(
        torch.tensor(
            rnd.randn(
                batch_dim_.dimension,
                enc_dim.dyn_size_ext.raw_tensor.max(),
                vocab_dim.dimension,
            ).astype(np.float32),
            device=_torch_default_device,
        ),
        dims=[batch_dim_, enc_dim, vocab_dim],
    )
    sizeof_float = 4
    print("logits shape:", logits.raw_tensor.shape)
    print("logits size:", logits.raw_tensor.numel() * sizeof_float, "bytes")
    print("dev:", logits.device)

    # Call this once because this will cache some things.
    # Exclude this part for the benchmark to have it fair.
    dims = batch_dims + [enc_dim]
    rf.sequence_mask(dims, device=logits.device)

    def _no_op_test():
        pass

    def _get_logits() -> Tensor:
        # Maybe do sth with logits, to better see effects of CUDA host-device synchronization.
        # return rf.matmul(logits, eye, reduce=vocab_dim)
        # return logits * 0.9 + 0.1
        return logits

    def _get_rf_pack_packed() -> torch.Tensor:
        logits_ = _get_logits()
        logits_packed, pack_dim = rf.pack_padded(logits_, dims=dims, enforce_sorted=False)  # [B * T, D]
        return logits_packed.raw_tensor

    def _get_rf_pack_padded_known_lens() -> torch.Tensor:
        logits_ = _get_logits()
        mask = rf.sequence_mask(dims, device=logits.device)
        assert mask.dims_set == set(dims)
        # Note: Already calculating out_dim here can trigger a more efficient calculation path in masked_select,
        # where we can avoid a CUDA host-device synchronization, e.g. in the PyTorch backend.
        # See https://github.com/rwth-i6/returnn/pull/1593.
        pack_dim = Dim(rf.num_elements_of_shape(dims), name="packed")
        logits_packed, _ = rf.masked_select(logits_, mask=mask, dims=dims, out_dim=pack_dim)
        return logits_packed.raw_tensor

    def _get_rf_pack_padded_no_known_lens() -> torch.Tensor:
        logits_ = _get_logits()
        mask = rf.sequence_mask(dims, device=logits.device)
        assert mask.dims_set == set(dims)
        logits_packed, pack_dim = rf.masked_select(logits_, mask=mask, dims=dims)
        return logits_packed.raw_tensor

    def _get_torch_masked_select_pack_padded() -> torch.Tensor:
        logits_ = _get_logits()
        # This was the old implementation of rf.pack_padded before https://github.com/rwth-i6/returnn/pull/1586.
        remaining_dims = [vocab_dim]
        tensor_templ_dims = dims + remaining_dims
        mask = rf.sequence_mask(dims, device=logits.device)
        in_raw = logits_.copy_compatible_to_dims_raw(tensor_templ_dims)
        mask_raw = mask.copy_compatible_to_dims_raw(tensor_templ_dims)
        out_raw = torch.masked_select(in_raw, mask_raw)
        remaining_shape = [d.get_dim_value() for d in remaining_dims]
        remaining_num_elements = numpy.prod(remaining_shape) if remaining_shape else 1
        assert out_raw.numel() % remaining_num_elements == 0
        flattened_num_elements = out_raw.numel() // remaining_num_elements
        out_raw = torch.reshape(out_raw, [flattened_num_elements] + remaining_shape)
        return out_raw

    def _get_naive_pack_padded() -> torch.Tensor:
        logits_ = _get_logits()
        tensor_templ_dims = dims + [vocab_dim]
        logits_raw = logits_.copy_compatible_to_dims_raw(tensor_templ_dims)
        enc_lens = enc_dim.dyn_size_ext.raw_tensor
        vocab_len = vocab_dim.dimension

        batch_tensors = []

        for b in range(logits_raw.shape[0]):
            enc_len = enc_lens[b]
            combined_len = enc_len
            logits_single = logits_raw[b, :enc_len]
            logits_single = torch.reshape(logits_single, (combined_len, vocab_len))
            batch_tensors.append(logits_single)

        return torch.cat(batch_tensors, dim=0)

    def _get_torch_pack_padded_sequence() -> torch.Tensor:
        logits_ = _get_logits()
        tensor_templ_dims = dims + [vocab_dim]
        logits_raw = logits_.copy_compatible_to_dims_raw(tensor_templ_dims)  # [B,T,D]
        enc_lens = enc_dim.dyn_size_ext.raw_tensor
        return torch.nn.utils.rnn.pack_padded_sequence(
            logits_raw,
            enc_lens,
            batch_first=True,
            # enforce_sorted=False is not totally equivalent to the other code here...
            enforce_sorted=False,
        ).data

    for f in [
        _no_op_test,  # test
        _get_logits,  # warmup dummy
        _get_rf_pack_packed,
        _get_rf_pack_padded_known_lens,
        _get_rf_pack_padded_no_known_lens,
        _get_torch_masked_select_pack_padded,
        _get_naive_pack_padded,
        _get_torch_pack_padded_sequence,
    ]:
        print("func:", f.__name__)
        t = Timer(stmt="func()", globals={"func": f})
        try:
            print(t.blocked_autorange(min_run_time=0.5))
        except torch.cuda.OutOfMemoryError:
            print("CUDA OOM")


def test_Data_copy_compatible_to_match_priority():
    feat_dim = Dim(2, name="feature")
    in_dim = feat_dim.copy(match_priority=1)
    assert in_dim == feat_dim and in_dim.match_priority > feat_dim.match_priority and in_dim is not feat_dim

    raw_np = numpy.arange(0, 2 * 2, dtype=numpy.float32).reshape((2, 2))
    raw = torch.tensor(raw_np)
    x = Tensor("x", [in_dim, feat_dim], "float32", raw_tensor=raw)

    # (in,out) -> (in,out) (noop)
    x_ = x.copy_compatible_to(Tensor("y", [in_dim, feat_dim], "float32"))
    assert len(x_.dims) == 2 and x_.dims[0] is in_dim and x_.dims[1] is feat_dim
    x_np = x_.raw_tensor.detach().numpy()
    numpy.testing.assert_equal(x_np, raw_np)

    # (in,out) -> (out,in)
    x_ = x.copy_compatible_to(Tensor("y", [feat_dim, in_dim], "float32"))
    assert len(x_.dims) == 2 and x_.dims[0] is feat_dim and x_.dims[1] is in_dim
    x_np = x_.raw_tensor.detach().numpy()
    numpy.testing.assert_equal(x_np, raw_np.transpose([1, 0]))

    # (out,in) -> (out,in) (noop)
    x_ = x_.copy_compatible_to(Tensor("y", [feat_dim, in_dim], "float32"))
    assert len(x_.dims) == 2 and x_.dims[0] is feat_dim and x_.dims[1] is in_dim
    x_np = x_.raw_tensor.detach().numpy()
    numpy.testing.assert_equal(x_np, raw_np.transpose([1, 0]))

    # (out,in) -> (in,out)
    x_ = x_.copy_compatible_to(Tensor("y", [in_dim, feat_dim], "float32"))
    assert len(x_.dims) == 2 and x_.dims[0] is in_dim and x_.dims[1] is feat_dim
    x_np = x_.raw_tensor.detach().numpy()
    numpy.testing.assert_equal(x_np, raw_np)


def test_Data_copy_compatible_to_dims_match_priority():
    feat_dim = Dim(2, name="feature")
    in_dim = feat_dim.copy(match_priority=1)
    assert in_dim == feat_dim and in_dim.match_priority > feat_dim.match_priority and in_dim is not feat_dim

    raw_np = numpy.arange(0, 2 * 2, dtype=numpy.float32).reshape((2, 2))
    raw = torch.tensor(raw_np)
    x = Tensor("x", [in_dim, feat_dim], "float32", raw_tensor=raw)

    # (in,out) -> (in,out) (noop)
    x_ = x.copy_compatible_to_dims([in_dim, feat_dim])
    assert len(x_.dims) == 2 and x_.dims[0] is in_dim and x_.dims[1] is feat_dim
    x_np = x_.raw_tensor.detach().numpy()
    numpy.testing.assert_equal(x_np, raw_np)

    # (in,out) -> (out,in)
    x_ = x.copy_compatible_to_dims([feat_dim, in_dim])
    assert len(x_.dims) == 2 and x_.dims[0] is feat_dim and x_.dims[1] is in_dim
    x_np = x_.raw_tensor.detach().numpy()
    numpy.testing.assert_equal(x_np, raw_np.transpose([1, 0]))

    # (out,in) -> (out,in) (noop)
    x_ = x_.copy_compatible_to_dims([feat_dim, in_dim])
    assert len(x_.dims) == 2 and x_.dims[0] is feat_dim and x_.dims[1] is in_dim
    x_np = x_.raw_tensor.detach().numpy()
    numpy.testing.assert_equal(x_np, raw_np.transpose([1, 0]))

    # (out,in) -> (in,out)
    x_ = x_.copy_compatible_to_dims([in_dim, feat_dim])
    assert len(x_.dims) == 2 and x_.dims[0] is in_dim and x_.dims[1] is feat_dim
    x_np = x_.raw_tensor.detach().numpy()
    numpy.testing.assert_equal(x_np, raw_np)


def test_Data_copy_tranpose_match_priority():
    feat_dim = Dim(2, name="feature")
    in_dim = feat_dim.copy(match_priority=1)
    assert in_dim == feat_dim and in_dim.match_priority > feat_dim.match_priority and in_dim is not feat_dim

    raw_np = numpy.arange(0, 2 * 2, dtype=numpy.float32).reshape((2, 2))
    raw = torch.tensor(raw_np)
    x = Tensor("x", [in_dim, feat_dim], "float32", raw_tensor=raw)

    # (in,out) -> (in,out) (noop)
    x_ = x.copy_transpose([in_dim, feat_dim])
    assert len(x_.dims) == 2 and x_.dims[0] is in_dim and x_.dims[1] is feat_dim
    x_np = x_.raw_tensor.detach().numpy()
    numpy.testing.assert_equal(x_np, raw_np)

    # (in,out) -> (out,in)
    x_ = x.copy_transpose([feat_dim, in_dim])
    assert len(x_.dims) == 2 and x_.dims[0] is feat_dim and x_.dims[1] is in_dim
    x_np = x_.raw_tensor.detach().numpy()
    numpy.testing.assert_equal(x_np, raw_np.transpose([1, 0]))

    # (out,in) -> (out,in) (noop)
    x_ = x_.copy_transpose([feat_dim, in_dim])
    assert len(x_.dims) == 2 and x_.dims[0] is feat_dim and x_.dims[1] is in_dim
    x_np = x_.raw_tensor.detach().numpy()
    numpy.testing.assert_equal(x_np, raw_np.transpose([1, 0]))

    # (out,in) -> (in,out)
    x_ = x_.copy_transpose([in_dim, feat_dim])
    assert len(x_.dims) == 2 and x_.dims[0] is in_dim and x_.dims[1] is feat_dim
    x_np = x_.raw_tensor.detach().numpy()
    numpy.testing.assert_equal(x_np, raw_np)


def test_forward_hook():
    from returnn.frontend import hooks
    from types import MethodType, FunctionType

    call_count = 0
    hook_call_count = 0

    class _Module(rf.Module):
        def __call__(self, x: Tensor) -> Tensor:
            nonlocal call_count
            call_count += 1
            return x * 0.5

    def _hook(module, args, kwargs, result):
        nonlocal hook_call_count
        assert isinstance(module, _Module) and module is mod
        assert isinstance(args, tuple) and len(args) == 1 and isinstance(args[0], Tensor)
        assert isinstance(kwargs, dict) and len(kwargs) == 0
        assert isinstance(result, Tensor)
        hook_call_count += 1
        return result * 2.0

    mod = _Module()
    time_dim = Dim(None, name="time")
    x = Tensor("x", [time_dim], "float32", raw_tensor=torch.range(0.0, 5.0))
    mod(x)
    assert call_count == 1 and hook_call_count == 0

    handle = mod.register_forward_hook(_hook)
    assert isinstance(handle, hooks.RemovableHandle)
    assert isinstance(mod.__call__, hooks.MethodWithHooks)
    assert isinstance(mod.__class__.__call__, hooks._CallWrapperClass)
    mod(x)
    assert call_count == 2 and hook_call_count == 1

    mod.__call__(x)
    assert call_count == 3 and hook_call_count == 2

    handle.remove()
    assert isinstance(mod.__call__, MethodType)
    assert isinstance(mod.__class__.__call__, FunctionType)
    mod(x)
    assert call_count == 4 and hook_call_count == 2

    mod.__call__(x)
    assert call_count == 5 and hook_call_count == 2


def test_convert_parameter_to_buffer():
    from returnn.torch.frontend.bridge import rf_module_to_pt_module
    from returnn.torch.util.module import convert_parameters_to_buffers

    # Replicate the case of Engine preload_from_files with init_for_train == "always".
    in_dim = Dim(2, name="in")
    out_dim = Dim(3, name="out")
    mod_rf = rf.Linear(in_dim, out_dim)
    mod_pt = rf_module_to_pt_module(mod_rf)
    convert_parameters_to_buffers(mod_pt, ["weight", "bias"], persistent=False)
    assert dict(mod_pt.named_parameters()) == {}
    assert set(dict(mod_pt.named_buffers()).keys()) == {"weight", "bias"}
    # The following type checks are implementation details,
    # and maybe we might change this behavior in the future.
    assert type(mod_pt.weight) is torch.nn.Parameter  # still same instance
    # After mod_pt.to(...), buffers will not be torch.nn.Parameter anymore.
    # Note, the test here is actually to run the to() method,
    # which runs our RFModuleAsPTModule._apply, because there was a bug, leading to:
    #   Linear[RF→PT]().weight is not a Parameter but Tensor
    mod_pt.to(torch.bfloat16)
    assert type(mod_pt.weight) is torch.Tensor


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
