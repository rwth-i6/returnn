"""
Tests for the packed monotonic RNN-T loss, see :mod:`returnn.torch.util.monotonic_rnnt`.
"""

from __future__ import annotations

import itertools
import sys

import torch

import _setup_test_env  # noqa
from returnn.torch.util.monotonic_rnnt import (
    cell_offsets,
    lattice_index,
    lattice_operands,
    monotonic_rnnt_loss,
    total_cells,
)


def _brute_force(logits: torch.Tensor, labels, blank: int) -> float:
    """
    Sums over every monotonic alignment explicitly, as the definition of the loss.

    :param logits: [T, U + 1, V] of one sequence
    :param labels: the U labels
    :param blank: blank index
    :return: the negative log likelihood
    """
    num_frames, num_prefix, _ = logits.shape
    num_labels = num_prefix - 1
    log_probs = torch.log_softmax(logits.double(), dim=-1)
    totals = []
    for emits in itertools.combinations(range(num_frames), num_labels):
        u = 0
        total = 0.0
        for t in range(num_frames):
            if t in emits:
                total += float(log_probs[t, u, labels[u]])
                u += 1
            else:
                total += float(log_probs[t, u, blank])
        totals.append(total)
    if not totals:
        return float("inf")
    return -float(torch.logsumexp(torch.tensor(totals, dtype=torch.float64), dim=0))


def _pack(per_seq_logits):
    """
    :param per_seq_logits: one [T, U + 1, V] tensor per sequence
    :return: the packed [cells, V] tensor the loss takes, frame index outer
    """
    return torch.cat([x.reshape(-1, x.shape[-1]) for x in per_seq_logits], dim=0)


def test_monotonic_rnnt_matches_the_sum_over_alignments():
    torch.manual_seed(42)
    vocab, blank = 5, 0
    cases = [(4, 2), (5, 0), (3, 3), (6, 2), (5, 4)]
    per_seq, labels, frame_lens, label_lens, want = [], [], [], [], []
    for num_frames, num_labels in cases:
        logits = torch.randn(num_frames, num_labels + 1, vocab)
        seq_labels = torch.randint(1, vocab, (num_labels,))
        if num_labels >= 2:
            seq_labels[1] = seq_labels[0]
        per_seq.append(logits)
        labels.append(seq_labels)
        frame_lens.append(num_frames)
        label_lens.append(num_labels)
        want.append(_brute_force(logits, seq_labels.tolist(), blank))

    labels_padded = torch.zeros(len(cases), max(label_lens) + 3, dtype=torch.int32)
    for i, seq_labels in enumerate(labels):
        labels_padded[i, : len(seq_labels)] = seq_labels

    got = monotonic_rnnt_loss(
        _pack(per_seq),
        labels_padded,
        torch.tensor(frame_lens, dtype=torch.int32),
        torch.tensor(label_lens, dtype=torch.int32),
        blank=blank,
    )
    torch.testing.assert_close(got.double(), torch.tensor(want, dtype=torch.float64), rtol=0, atol=1e-6)


def test_lattice_index_refuses_a_batch_above_the_capacity():
    """
    A capacity below the batch would put the cells past it onto the last sequence, cut short in silence, and
    below the sum of the other sequences the last span turned negative, which crashed the process. Cells past
    the batch's own sum still land on the last sequence beyond its end, where the loss masks them.
    """
    frame_lens = torch.tensor([4, 3, 5], dtype=torch.int32)
    label_lens = torch.tensor([2, 1, 0], dtype=torch.int32)
    total = total_cells(frame_lens, label_lens)

    for capacity in (total - 1, total - 6):
        try:
            lattice_index(frame_lens, label_lens, capacity)
        except RuntimeError as exc:
            assert "cells" in str(exc), exc
        else:
            raise AssertionError(f"a capacity of {capacity} below {total} cells was accepted")

    seq, frame, _prefix = lattice_index(frame_lens, label_lens, total + 4)
    assert seq[total:].tolist() == [2] * 4, seq
    assert bool((frame[total:] >= 5).all()), frame


def test_lattice_operands_match_the_per_sequence_loop():
    torch.manual_seed(11)
    cases = [(9, 4), (12, 0), (7, 7), (5, 2)]
    max_frames, max_labels = max(t for t, _u in cases), max(u for _t, u in cases)
    enc = torch.randn(len(cases), max_frames, 6)
    pred = torch.randn(len(cases), max_labels + 1, 3)
    frame_lens = torch.tensor([t for t, _u in cases], dtype=torch.int32)
    label_lens = torch.tensor([u for _t, u in cases], dtype=torch.int32)

    want_enc, want_pred = [], []
    for i, (num_frames, num_labels) in enumerate(cases):
        rows, cols = num_frames, num_labels + 1
        want_enc.append(enc[i, :rows].unsqueeze(1).expand(rows, cols, 6).reshape(rows * cols, 6))
        want_pred.append(pred[i, :cols].unsqueeze(0).expand(rows, cols, 3).reshape(rows * cols, 3))
    want_enc, want_pred = torch.cat(want_enc), torch.cat(want_pred)

    total = total_cells(frame_lens, label_lens)
    assert total == want_enc.shape[0], (total, want_enc.shape)
    got_enc, got_pred = lattice_operands(enc, pred, frame_lens, label_lens, total + 7)
    torch.testing.assert_close(got_enc[:total], want_enc)
    torch.testing.assert_close(got_pred[:total], want_pred)


def test_monotonic_rnnt_gradient_matches_finite_differences():
    torch.manual_seed(7)
    vocab, blank = 4, 0
    num_frames, num_labels = 4, 2
    logits = torch.randn(num_frames, num_labels + 1, vocab, dtype=torch.float64, requires_grad=True)
    seq_labels = torch.tensor([1, 3], dtype=torch.int32)
    args = (
        seq_labels.unsqueeze(0),
        torch.tensor([num_frames], dtype=torch.int32),
        torch.tensor([num_labels], dtype=torch.int32),
    )
    torch.autograd.gradcheck(
        lambda x: monotonic_rnnt_loss(x.reshape(-1, vocab), *args, blank=blank),
        (logits,),
        eps=1e-6,
        atol=1e-6,
    )


def test_monotonic_rnnt_on_cuda_matches_the_reference():
    """
    What the kernels see under capture, a buffer and a recursion above the batch's own cells and frames,
    a sequence with more labels than frames and a batch without any label, each against the cpu path.
    """
    if not torch.cuda.is_available():
        import unittest

        raise unittest.SkipTest("no cuda")
    torch.manual_seed(3)
    vocab, blank, slack = 37, 0, 13
    for cases in ([(9, 4), (12, 0), (7, 7), (5, 2), (3, 5)], [(4, 0), (6, 0)]):
        per_seq, labels, frame_lens, label_lens = [], [], [], []
        for num_frames, num_labels in cases:
            per_seq.append(torch.randn(num_frames, num_labels + 1, vocab))
            labels.append(torch.randint(1, vocab, (num_labels,)))
            frame_lens.append(num_frames)
            label_lens.append(num_labels)
        labels_padded = torch.zeros(len(cases), max(label_lens), dtype=torch.int32)
        for i, seq_labels in enumerate(labels):
            labels_padded[i, : len(seq_labels)] = seq_labels
        frame_lens_t = torch.tensor(frame_lens, dtype=torch.int32)
        label_lens_t = torch.tensor(label_lens, dtype=torch.int32)
        packed = torch.cat([_pack(per_seq), torch.randn(slack, vocab)])
        grad_out = torch.randn(len(cases))

        results = []
        for device in ("cpu", "cuda"):
            logits = packed.to(device).clone().requires_grad_()
            loss = monotonic_rnnt_loss(
                logits,
                labels_padded.to(device),
                frame_lens_t.to(device),
                label_lens_t.to(device),
                blank=blank,
                max_frames=max(frame_lens) + 2,
            )
            loss.backward(grad_out.to(device))
            results.append((loss.detach().cpu(), logits.grad.detach().cpu()))
        torch.testing.assert_close(results[1][0], results[0][0], rtol=1e-5, atol=1e-5, msg=str(cases))
        torch.testing.assert_close(results[1][1], results[0][1], rtol=1e-4, atol=1e-5, msg=str(cases))
        if cases[-1][1] > cases[-1][0]:
            assert float(results[1][0][-1]) == 0.0, results[1][0]


def test_monotonic_rnnt_handles_degenerate_batches():
    torch.manual_seed(3)
    vocab, blank = 6, 0

    frame_lens = torch.tensor([0, 0], dtype=torch.int32)
    label_lens = torch.tensor([5, 7], dtype=torch.int32)
    got = monotonic_rnnt_loss(
        torch.randn(0, vocab), torch.randint(1, vocab, (2, 7), dtype=torch.int32), frame_lens, label_lens,
        blank=blank, max_frames=3,
    )
    torch.testing.assert_close(got, torch.zeros(2))

    frame_lens = torch.tensor([4, 3], dtype=torch.int32)
    label_lens = torch.tensor([0, 0], dtype=torch.int32)
    logits = torch.randn(7, vocab)
    got = monotonic_rnnt_loss(
        logits, torch.zeros((2, 0), dtype=torch.int32), frame_lens, label_lens, blank=blank, max_frames=4
    )
    blank_lp = torch.log_softmax(logits.double(), dim=-1)[:, blank]
    want = torch.tensor([-float(blank_lp[:4].sum()), -float(blank_lp[4:].sum())])
    torch.testing.assert_close(got.double(), want.double(), rtol=0, atol=1e-5)


def test_cell_stats_normalizer_survives_a_block_of_minus_infinity():
    if not torch.cuda.is_available():
        import unittest

        raise unittest.SkipTest("no cuda")
    from returnn.torch.util.monotonic_rnnt_triton import cell_stats

    torch.manual_seed(2)
    vocab, blank = 1025, 0
    logits = torch.randn(3, vocab, device="cuda")
    logits[1, :1024] = float("-inf")
    next_label = torch.tensor([7, 1024, 3], dtype=torch.int64, device="cuda")

    lse, blank_lp, label_lp = cell_stats(logits, next_label, blank)
    log_probs = torch.log_softmax(logits.double(), dim=-1)
    want_lse = torch.logsumexp(logits.double(), dim=-1)
    torch.testing.assert_close(lse.double(), want_lse, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(blank_lp.double(), log_probs[:, blank], rtol=1e-5, atol=1e-5)
    rows = torch.arange(3, device="cuda")
    torch.testing.assert_close(label_lp.double(), log_probs[rows, next_label], rtol=1e-5, atol=1e-5)


def test_monotonic_rnnt_traces_under_aot():
    if not torch.cuda.is_available():
        import unittest

        raise unittest.SkipTest("no cuda")
    from functorch.compile import aot_function, nop

    torch.manual_seed(5)
    vocab, blank, batch, frames, num_labels = 32, 0, 3, 8, 3
    frame_lens = torch.full((batch,), frames, dtype=torch.int32, device="cuda")
    label_lens = torch.full((batch,), num_labels, dtype=torch.int32, device="cuda")
    labels = torch.randint(1, vocab, (batch, num_labels), dtype=torch.int32, device="cuda")
    packed = torch.randn(batch * frames * (num_labels + 1), vocab, device="cuda")
    grad_out = torch.ones(batch, device="cuda")

    def loss_of(x, lab, flens, ulens):
        return monotonic_rnnt_loss(x, lab, flens, ulens, blank=blank, max_frames=frames)

    want = []
    for func in (loss_of, aot_function(loss_of, fw_compiler=nop, bw_compiler=nop)):
        x = packed.clone().requires_grad_()
        loss = func(x, labels, frame_lens, label_lens)
        loss.backward(grad_out)
        want.append((loss.detach().clone(), x.grad.clone()))
    torch.testing.assert_close(want[1][0], want[0][0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(want[1][1], want[0][1], rtol=1e-4, atol=1e-6)


def test_monotonic_rnnt_takes_the_logits_in_their_own_dtype():
    """
    The kernels read every logit anyway, so they convert on load and no float32 copy of the lattice is made.
    Such a copy would double the traffic of the largest tensor in the step and, worse, make autograd carry
    its gradient back through the cast, which is another pass over the whole lattice.
    """
    if not torch.cuda.is_available():
        print("   (skipped, needs cuda)")
        return
    cells, vocab = 4096, 1024
    frame_lens = torch.tensor([32, 32], dtype=torch.int32, device="cuda")
    label_lens = torch.tensor([63, 63], dtype=torch.int32, device="cuda")
    labels = torch.randint(0, vocab - 1, (2, 63), dtype=torch.int32, device="cuda")
    logits = torch.randn(cells, vocab, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    loss = monotonic_rnnt_loss(logits, labels, frame_lens, label_lens, blank=vocab - 1, max_frames=32)
    loss.sum().backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before

    assert logits.grad.dtype == torch.bfloat16, logits.grad.dtype
    lattice_fp32 = cells * vocab * 4
    assert peak < lattice_fp32, f"peak {peak} implies a float32 copy of the lattice ({lattice_fp32} bytes)"

    reference = logits.detach().float().requires_grad_()
    want = monotonic_rnnt_loss(reference, labels, frame_lens, label_lens, blank=vocab - 1, max_frames=32)
    want.sum().backward()
    torch.testing.assert_close(loss, want, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(logits.grad.float(), reference.grad, rtol=2e-2, atol=2e-5)


def test_monotonic_rnnt_refuses_a_recursion_shorter_than_a_sequence():
    """
    A recursion cut short scored the sequence at the sentinel on cpu and gave a partial score on cuda that
    looked like any other loss, so the frames a caller declares have to cover every sequence.
    """
    torch.manual_seed(1)
    logits = torch.randn(12, 5)
    labels = torch.tensor([[1, 2]], dtype=torch.int32)
    frame_lens, label_lens = torch.tensor([4], dtype=torch.int32), torch.tensor([2], dtype=torch.int32)
    fine = monotonic_rnnt_loss(logits, labels, frame_lens, label_lens, blank=0, max_frames=4)
    assert torch.isfinite(fine).all(), fine

    try:
        monotonic_rnnt_loss(logits, labels, frame_lens, label_lens, blank=0, max_frames=3)
    except RuntimeError as exc:
        assert "frames" in str(exc), exc
    else:
        raise AssertionError("a recursion shorter than the sequence was accepted")


def test_monotonic_rnnt_refuses_labels_outside_the_vocabulary():
    """
    The cell kernels index the row by blank and by label unchecked, so a stray id reads and writes out of
    bounds on cuda, and both ids are checked before any kernel runs.
    """
    torch.manual_seed(1)
    vocab = 5
    logits = torch.randn(12, vocab)
    frame_lens, label_lens = torch.tensor([4], dtype=torch.int32), torch.tensor([2], dtype=torch.int32)
    for labels, blank, error in (
        ([[1, 2]], vocab, AssertionError),
        ([[1, vocab]], 0, RuntimeError),
        ([[-1, 2]], 0, RuntimeError),
    ):
        try:
            monotonic_rnnt_loss(
                logits, torch.tensor(labels, dtype=torch.int32), frame_lens, label_lens, blank=blank, max_frames=4
            )
        except error as exc:
            assert "vocabulary" in str(exc), exc
        else:
            raise AssertionError(f"labels {labels} with blank {blank} were accepted")


def test_backward_scan_leaves_an_unalignable_sequence_alone():
    """
    A sequence with more labels than frames has no alignment, its normalizer sits at the sentinel and every
    cell's posterior came out as exp(0), which only a caller's zero weight hid. The sweep gives zero itself.
    """
    if not torch.cuda.is_available():
        import unittest

        raise unittest.SkipTest("no cuda")
    from returnn.torch.util.monotonic_rnnt_triton import backward_scan, forward_scan

    torch.manual_seed(4)
    frame_lens = torch.tensor([3, 4], dtype=torch.int32, device="cuda")
    label_lens = torch.tensor([5, 2], dtype=torch.int32, device="cuda")
    offsets, cells = cell_offsets(frame_lens, label_lens)
    log_probs = torch.log_softmax(torch.randn(int(cells.sum()), 3, device="cuda"), dim=-1)
    blank_lp, label_lp = log_probs[:, 0].contiguous(), log_probs[:, 1].contiguous()

    total, alpha = forward_scan(blank_lp, label_lp, offsets, frame_lens, label_lens, 4, 6)
    blank_grad, label_grad = backward_scan(
        blank_lp, label_lp, offsets, frame_lens, label_lens, alpha, total, torch.ones(2, device="cuda")
    )
    first = int(cells[0])
    assert not blank_grad[:first].any() and not label_grad[:first].any(), (blank_grad[:first], label_grad[:first])
    assert torch.isfinite(blank_grad[first:]).all() and float(blank_grad[first:].abs().sum()) > 0.0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        globals()[sys.argv[1]]()
    else:
        for name, func in sorted(globals().items()):
            if name.startswith("test_"):
                print(f"-- {name}")
                func()
        print("all passed")
