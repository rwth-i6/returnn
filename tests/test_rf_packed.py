"""
Tests for returnn.frontend._packed_backend (packed / ragged tensor storage).

Each test compares the packed path against the same computation on padded storage,
on all non-padded frames.
"""

from __future__ import annotations

import _setup_test_env  # noqa
import sys
import unittest
from typing import Tuple

import numpy
import torch

from returnn.util import better_exchook
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend import _packed_backend as packed


def _make_input(
    *, batch_size: int = 2, seq_lens=(5, 3), feat: int = 4, seed: int = 42
) -> Tuple[Tensor, Dim, Dim, Dim]:
    batch_dim = Dim(batch_size, name="batch")
    time_dim = Dim(
        Tensor("time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor(list(seq_lens), dtype=torch.int32))
    )
    feat_dim = Dim(feat, name="feat")
    x = Tensor("x", dims=[batch_dim, time_dim, feat_dim], dtype="float32")
    raw = torch.randn(batch_size, max(seq_lens), feat, generator=torch.Generator().manual_seed(seed))
    x.raw_tensor = raw
    return x, batch_dim, time_dim, feat_dim


def _assert_equal_non_padded(actual: Tensor, expected: Tensor, batch_dim: Dim, time_dim: Dim, **kwargs):
    """compare on all non-padded frames. actual can have packed storage."""
    actual = packed.unpack(actual)
    actual = actual.copy_compatible_to_dims(expected.dims)
    mask = rf.sequence_mask([batch_dim, time_dim]).copy_compatible_to_dims([batch_dim, time_dim]).raw_tensor.numpy()
    a = actual.raw_tensor.detach().numpy()
    e = expected.raw_tensor.detach().numpy()
    numpy.testing.assert_allclose(a[mask], e[mask], **{"rtol": 1e-5, "atol": 1e-6, **kwargs})


def test_pack_auto_dims():
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    xp = packed.pack(x)  # no dims given: auto = (batch, time)
    assert packed.is_packed(xp)
    assert xp.dims == x.dims
    raw = xp.raw_tensor
    assert raw.orig_dims == (batch_dim, time_dim)
    assert raw.inner.dims[0] == raw.packed_dim
    assert raw.packed_dim.get_dim_value() == sum([5, 3])
    _assert_equal_non_padded(xp, x, batch_dim, time_dim)


def test_elementwise():
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    xp = packed.pack(x)
    out_p = rf.relu(xp) * 2.0 + 1.0
    assert packed.is_packed(out_p)
    _assert_equal_non_padded(out_p, rf.relu(x) * 2.0 + 1.0, batch_dim, time_dim)


def test_linear():
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    out_dim = Dim(3, name="out")
    layer = rf.Linear(feat_dim, out_dim)  # with bias: also covers packed + plain combine
    xp = packed.pack(x)
    out_p = layer(xp)
    assert packed.is_packed(out_p)  # matmul over feat + bias add must stay packed
    _assert_equal_non_padded(out_p, layer(x), batch_dim, time_dim)


def test_layer_norm():
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    layer = rf.LayerNorm(feat_dim)
    xp = packed.pack(x)
    out_p = layer(xp)
    assert packed.is_packed(out_p)  # statistics are over feat only, must stay packed
    _assert_equal_non_padded(out_p, layer(x), batch_dim, time_dim)


def test_output_block_log_softmax():
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    vocab_dim = Dim(11, name="vocab")
    layer = rf.Linear(feat_dim, vocab_dim)
    xp = packed.pack(x)
    out_p = rf.log_softmax(layer(xp), axis=vocab_dim)
    assert packed.is_packed(out_p)  # softmax over vocab must stay packed
    _assert_equal_non_padded(out_p, rf.log_softmax(layer(x), axis=vocab_dim), batch_dim, time_dim)


def test_reduce_mean_over_packed_dims():
    # e.g. the mean loss over (batch, time): reduces the packed dim directly, no masking needed.
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    xp = packed.pack(x)
    out_p = rf.reduce_mean(xp, axis=[batch_dim, time_dim])
    assert not packed.is_packed(out_p)  # packed dims fully reduced
    out_ref = rf.reduce_mean(x, axis=[batch_dim, time_dim])  # masked mean on padded storage
    numpy.testing.assert_allclose(
        out_p.raw_tensor.detach().numpy(), out_ref.raw_tensor.detach().numpy(), rtol=1e-5, atol=1e-6
    )


def test_reduce_over_time_fallback():
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    xp = packed.pack(x)
    out_p = rf.reduce_max(xp, axis=time_dim)  # partial packed reduce: unpack fallback
    out_ref = rf.reduce_max(x, axis=time_dim)
    out_p = out_p.copy_compatible_to_dims(out_ref.dims)
    numpy.testing.assert_allclose(
        out_p.raw_tensor.detach().numpy(), out_ref.raw_tensor.detach().numpy(), rtol=1e-5, atol=1e-6
    )


def test_conformer():
    """
    The goal test: a full Conformer forward pass on packed input matches the padded path.

    Currently most of the Conformer internals (conv subsampling, attention) go through
    the unpack fallback and decay to padded storage --
    this test establishes end-to-end correctness first;
    packed attention (flash-varlen) and packed conv come incrementally.
    """
    rf.select_backend_torch()
    from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

    x, batch_dim, time_dim, in_dim = _make_input(batch_size=2, seq_lens=(11, 7), feat=7, seed=3)
    with rf.set_default_device_ctx("cpu"):
        rf.set_random_seed(17)
        model = ConformerEncoder(
            in_dim,
            Dim(14, name="enc"),
            ff_dim=Dim(17, name="ff"),
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[Dim(8, name="conv1"), Dim(8, name="conv2")],
                filter_sizes=[(3, 3), (3, 3)],
                pool_sizes=[(2, 1), (2, 1)],
            ),
            num_heads=2,
            num_layers=2,
        )
        out_ref, out_spatial_dim = model(x, in_spatial_dim=time_dim)
        xp = packed.pack(x)
        out_p, out_spatial_dim_p = model(xp, in_spatial_dim=time_dim)
    assert out_spatial_dim == out_spatial_dim_p
    _assert_equal_non_padded(packed.unpack(out_p), out_ref, batch_dim, out_spatial_dim, rtol=1e-4, atol=1e-5)


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
