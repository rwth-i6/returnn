"""
Tests for the packed monotonic RNN-T lattice, see :func:`returnn.frontend._packed_backend.monotonic_rnnt_lattice`.
"""

from __future__ import annotations

import sys

import torch

import _setup_test_env  # noqa
import returnn.frontend as rf
from returnn.frontend._packed_backend import monotonic_rnnt_lattice
from returnn.tensor import Dim, Tensor
from returnn.torch.util.monotonic_rnnt import lattice_operands, total_cells


_CASES = [(9, 4), (12, 0), (7, 7), (5, 2)]


def _batch():
    """
    :return: (enc, pred, enc_time, prefix_dim, frame_lens, label_lens) with the encoder and the predictor padded
    """
    torch.manual_seed(11)
    rf.select_backend_torch()
    frames = torch.tensor([t for t, _u in _CASES], dtype=torch.int32)
    prefixes = torch.tensor([u + 1 for _t, u in _CASES], dtype=torch.int32)
    num_seqs = len(_CASES)
    batch = Dim(num_seqs, name="batch")
    enc_time = Dim(
        Tensor("enc_lens", dims=[batch], dtype="int32", raw_tensor=frames), name="enc_time"
    )
    prefix_dim = Dim(
        Tensor("prefix_lens", dims=[batch], dtype="int32", raw_tensor=prefixes), name="prefixes"
    )
    enc_feat, pred_feat = Dim(6, name="enc_feat"), Dim(3, name="pred_feat")
    enc = Tensor(
        "enc",
        dims=[batch, enc_time, enc_feat],
        dtype="float32",
        raw_tensor=torch.randn(num_seqs, int(frames.max()), 6),
        feature_dim=enc_feat,
    )
    pred = Tensor(
        "pred",
        dims=[batch, prefix_dim, pred_feat],
        dtype="float32",
        raw_tensor=torch.randn(num_seqs, int(prefixes.max()), 3),
        feature_dim=pred_feat,
    )
    return enc, pred, enc_time, prefix_dim, frames, prefixes - 1


def test_monotonic_rnnt_lattice_matches_the_padded_gather():
    enc, pred, enc_time, prefix_dim, frame_lens, label_lens = _batch()
    cells = total_cells(frame_lens, label_lens)
    want_enc, want_pred = lattice_operands(enc.raw_tensor, pred.raw_tensor, frame_lens, label_lens, cells)

    for packed in (False, True):
        enc_in = rf.pack(enc, dims=[enc.dims[0], enc_time]) if packed else enc
        enc_cells, pred_cells, lattice_time = monotonic_rnnt_lattice(
            enc_in, pred, enc_spatial_dim=enc_time, prefix_dim=prefix_dim, cells_bound=cells
        )
        got_enc = enc_cells.raw_tensor.inner.raw_tensor
        got_pred = pred_cells.raw_tensor.inner.raw_tensor
        torch.testing.assert_close(got_enc, want_enc, msg=f"encoder cells differ, packed={packed}")
        torch.testing.assert_close(got_pred, want_pred, msg=f"predictor cells differ, packed={packed}")
        sizes = lattice_time.get_dyn_size_ext_for_device("cpu").raw_tensor
        torch.testing.assert_close(
            sizes.long(), (frame_lens.long() * (label_lens.long() + 1)), msg=f"lattice sizes, packed={packed}"
        )


def test_monotonic_rnnt_lattice_keeps_the_real_cells_under_a_capacity():
    enc, pred, enc_time, prefix_dim, frame_lens, label_lens = _batch()
    cells = total_cells(frame_lens, label_lens)
    want_enc, want_pred = lattice_operands(enc.raw_tensor, pred.raw_tensor, frame_lens, label_lens, cells)

    enc_cells, pred_cells, _lattice_time = monotonic_rnnt_lattice(
        enc, pred, enc_spatial_dim=enc_time, prefix_dim=prefix_dim, cells_bound=cells + 13
    )
    torch.testing.assert_close(enc_cells.raw_tensor.inner.raw_tensor[:cells], want_enc)
    torch.testing.assert_close(pred_cells.raw_tensor.inner.raw_tensor[:cells], want_pred)


def test_monotonic_rnnt_lattice_refuses_a_batch_above_the_capacity():
    """
    A batch whose cells exceed the capacity would silently lose the tail of its last sequence,
    the lattice index clamps the cells past the buffer onto its last cell. That is a wrong batcher
    cost, and it has to fail loudly rather than train on a truncated lattice.
    """
    enc, pred, enc_time, prefix_dim, frame_lens, label_lens = _batch()
    cells = total_cells(frame_lens, label_lens)

    try:
        monotonic_rnnt_lattice(enc, pred, enc_spatial_dim=enc_time, prefix_dim=prefix_dim, cells_bound=cells - 1)
    except RuntimeError as exc:
        assert "cells" in str(exc), exc
    else:
        raise AssertionError("a lattice above its capacity was built without an error")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        globals()[sys.argv[1]]()
    else:
        for name, func in sorted(globals().items()):
            if name.startswith("test_"):
                print(f"-- {name}")
                func()
        print("all passed")
