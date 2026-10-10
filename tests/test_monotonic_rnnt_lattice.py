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
    enc_time = Dim(Tensor("enc_lens", dims=[batch], dtype="int32", raw_tensor=frames), name="enc_time")
    prefix_dim = Dim(Tensor("prefix_lens", dims=[batch], dtype="int32", raw_tensor=prefixes), name="prefixes")
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


def _cell_operands(enc: torch.Tensor, pred: torch.Tensor, frame_lens: torch.Tensor, label_lens: torch.Tensor):
    """
    :return: the encoder frame and the predictor state of every lattice cell, by a loop over the sequences
    """
    enc_cells, pred_cells = [], []
    for b, (num_frames, num_labels) in enumerate(zip(frame_lens.tolist(), label_lens.tolist())):
        for t in range(num_frames):
            for u in range(num_labels + 1):
                enc_cells.append(enc[b, t])
                pred_cells.append(pred[b, u])
    return torch.stack(enc_cells), torch.stack(pred_cells)


def test_monotonic_rnnt_lattice_matches_the_padded_gather():
    enc, pred, enc_time, prefix_dim, frame_lens, label_lens = _batch()
    want_enc, want_pred = _cell_operands(enc.raw_tensor, pred.raw_tensor, frame_lens, label_lens)
    cells = want_enc.shape[0]

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
    want_enc, want_pred = _cell_operands(enc.raw_tensor, pred.raw_tensor, frame_lens, label_lens)
    cells = want_enc.shape[0]

    enc_cells, pred_cells, _lattice_time = monotonic_rnnt_lattice(
        enc, pred, enc_spatial_dim=enc_time, prefix_dim=prefix_dim, cells_bound=cells + 13
    )
    torch.testing.assert_close(enc_cells.raw_tensor.inner.raw_tensor[:cells], want_enc)
    torch.testing.assert_close(pred_cells.raw_tensor.inner.raw_tensor[:cells], want_pred)


def test_monotonic_rnnt_lattice_takes_an_operand_without_a_feature_dim():
    """
    A joint that reads more than one frame per cell, a whole chunk of them say, needs to know which row of
    the lattice a cell is in rather than one frame, so the operand is the index of the row itself.
    """
    _enc, pred, enc_time, prefix_dim, frame_lens, label_lens = _batch()
    batch = pred.dims[0]
    rows_dim = Dim(int(frame_lens.max()), name="rows")
    rows = Tensor(
        "rows",
        dims=[batch, enc_time],
        dtype="int32",
        raw_tensor=torch.arange(int(frame_lens.max()), dtype=torch.int32).expand(len(_CASES), -1).contiguous(),
        sparse_dim=rows_dim,
    )
    cells = int((frame_lens.long() * (label_lens.long() + 1)).sum())
    want = torch.cat([torch.arange(t, dtype=torch.int32).repeat_interleave(u + 1) for t, u in _CASES])

    row_cells, pred_cells, _lattice_time = monotonic_rnnt_lattice(
        rows, pred, enc_spatial_dim=enc_time, prefix_dim=prefix_dim, cells_bound=cells
    )
    assert row_cells.dims == pred_cells.dims[:-1], (row_cells, pred_cells)
    assert row_cells.sparse_dim == rows_dim, row_cells
    torch.testing.assert_close(row_cells.raw_tensor.inner.raw_tensor, want)


def test_monotonic_rnnt_loss_returns_the_log_probs_of_the_lattice():
    enc, pred, enc_time, prefix_dim, frame_lens, label_lens = _batch()
    batch, blank = enc.dims[0], 0
    vocab = Dim(5, name="vocab")
    labels_time = Dim(Tensor("label_lens", dims=[batch], dtype="int32", raw_tensor=label_lens), name="labels")
    labels_raw = torch.randint(1, vocab.dimension, (len(_CASES), int(label_lens.max())), dtype=torch.int32)
    labels = Tensor("labels", dims=[batch, labels_time], dtype="int32", raw_tensor=labels_raw, sparse_dim=vocab)
    enc_weight, pred_weight = torch.randn(enc.feature_dim.dimension, 5), torch.randn(pred.feature_dim.dimension, 5)

    def joint(enc_cells: Tensor, pred_cells: Tensor) -> Tensor:
        enc_w = rf.convert_to_tensor(enc_weight, dims=[enc.feature_dim, vocab])
        pred_w = rf.convert_to_tensor(pred_weight, dims=[pred.feature_dim, vocab])
        enc_part = rf.matmul(enc_cells, enc_w, reduce=enc.feature_dim)
        return enc_part + rf.matmul(pred_cells, pred_w, reduce=pred.feature_dim)

    # the joint at every position of the padded lattice, no label edge where the prefix is complete
    log_probs = torch.log_softmax(
        (enc.raw_tensor @ enc_weight).unsqueeze(2) + (pred.raw_tensor @ pred_weight).unsqueeze(1), dim=-1
    )
    frames = torch.arange(enc.raw_tensor.shape[1]).view(1, -1, 1)
    prefixes = torch.arange(pred.raw_tensor.shape[1]).view(1, 1, -1)
    inside = (frames < frame_lens.view(-1, 1, 1)) & (prefixes <= label_lens.view(-1, 1, 1))
    next_label = labels_raw.long().gather(1, prefixes[0].clamp(max=labels_raw.shape[1] - 1).expand(len(_CASES), -1))
    label_log_probs = log_probs.gather(-1, next_label.unsqueeze(1).expand(-1, frames.shape[1], -1).unsqueeze(-1))
    want_blank = torch.where(inside, log_probs[..., blank], float("-inf"))
    want_label = torch.where(inside & (prefixes < label_lens.view(-1, 1, 1)), label_log_probs[..., 0], float("-inf"))

    for packed in (False, True):
        enc_in = rf.pack(enc, dims=[batch, enc_time]) if packed else enc
        _loss, blank_lp, label_lp = rf.monotonic_rnnt_loss(
            enc=enc_in,
            pred=pred,
            enc_spatial_dim=enc_time,
            prefix_dim=prefix_dim,
            labels=labels,
            labels_spatial_dim=labels_time,
            joint=joint,
            blank_index=blank,
            return_log_probs=True,
        )
        for got, want in ((blank_lp, want_blank), (label_lp, want_label)):
            assert got.dims == (batch, enc_time, prefix_dim), got
            torch.testing.assert_close(got.raw_tensor, want, msg=f"{got.name}, packed={packed}")


def test_monotonic_rnnt_lattice_refuses_a_batch_above_the_capacity():
    """
    A batch whose cells exceed the capacity would silently lose the tail of its last sequence.
    That is a wrong batcher cost, and it has to fail loudly rather than train on a truncated lattice.
    """
    enc, pred, enc_time, prefix_dim, frame_lens, label_lens = _batch()
    cells = int((frame_lens.long() * (label_lens.long() + 1)).sum())

    try:
        monotonic_rnnt_lattice(enc, pred, enc_spatial_dim=enc_time, prefix_dim=prefix_dim, cells_bound=cells - 1)
    except AssertionError as exc:
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
