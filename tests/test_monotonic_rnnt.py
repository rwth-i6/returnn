"""
Tests for the packed monotonic RNN-T loss, see :mod:`returnn.torch.util.monotonic_rnnt`.
"""

from __future__ import annotations

import itertools
import sys

import torch

import _setup_test_env  # noqa
from returnn.torch.util.monotonic_rnnt import monotonic_rnnt_loss


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
        max_frames=max(frame_lens),
    )
    torch.testing.assert_close(got.double(), torch.tensor(want, dtype=torch.float64), rtol=0, atol=1e-6)


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
        lambda x: monotonic_rnnt_loss(x.reshape(-1, vocab), *args, blank=blank, max_frames=num_frames),
        (logits,),
        eps=1e-6,
        atol=1e-6,
    )


def test_monotonic_rnnt_on_cuda_matches_the_reference():
    if not torch.cuda.is_available():
        import unittest

        raise unittest.SkipTest("no cuda")
    torch.manual_seed(3)
    vocab, blank = 37, 0
    cases = [(9, 4), (12, 0), (7, 7), (5, 2)]
    per_seq, labels, frame_lens, label_lens = [], [], [], []
    for num_frames, num_labels in cases:
        per_seq.append(torch.randn(num_frames, num_labels + 1, vocab))
        seq_labels = torch.randint(1, vocab, (num_labels,))
        labels.append(seq_labels)
        frame_lens.append(num_frames)
        label_lens.append(num_labels)
    max_labels = max(label_lens) or 1
    labels_padded = torch.zeros(len(cases), max_labels, dtype=torch.int32)
    for i, seq_labels in enumerate(labels):
        labels_padded[i, : len(seq_labels)] = seq_labels
    frame_lens_t = torch.tensor(frame_lens, dtype=torch.int32)
    label_lens_t = torch.tensor(label_lens, dtype=torch.int32)
    packed = _pack(per_seq)
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
            max_frames=max(frame_lens),
        )
        loss.backward(grad_out.to(device))
        results.append((loss.detach().cpu(), logits.grad.detach().cpu()))
    torch.testing.assert_close(results[1][0], results[0][0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(results[1][1], results[0][1], rtol=1e-4, atol=1e-5)


def test_monotonic_rnnt_handles_degenerate_batches():
    torch.manual_seed(3)
    vocab, blank = 6, 0

    frame_lens = torch.tensor([0, 0], dtype=torch.int32)
    label_lens = torch.tensor([5, 7], dtype=torch.int32)
    got = monotonic_rnnt_loss(
        torch.randn(0, vocab),
        torch.randint(1, vocab, (2, 7), dtype=torch.int32),
        frame_lens,
        label_lens,
        blank=blank,
        max_frames=3,
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


def test_cell_stats_normalizer_survives_minus_infinity():
    if not torch.cuda.is_available():
        import unittest

        raise unittest.SkipTest("no cuda")
    from returnn.torch.util.monotonic_rnnt_triton import cell_stats

    torch.manual_seed(2)
    vocab, blank = 1025, 0
    logits = torch.randn(4, vocab, device="cuda")
    logits[1, :1024] = float("-inf")
    # A row without any finite logit has no probability mass, so its log probabilities are minus infinity, not nan.
    logits[3] = float("-inf")
    next_label = torch.tensor([7, 1024, 3, 5], dtype=torch.int64, device="cuda")

    lse, blank_lp, label_lp = cell_stats(logits, next_label, blank)
    log_probs = torch.log_softmax(logits[:3].double(), dim=-1)
    want_lse = torch.logsumexp(logits[:3].double(), dim=-1)
    torch.testing.assert_close(lse[:3].double(), want_lse, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(blank_lp[:3].double(), log_probs[:, blank], rtol=1e-5, atol=1e-5)
    rows = torch.arange(3, device="cuda")
    torch.testing.assert_close(label_lp[:3].double(), log_probs[rows, next_label[:3]], rtol=1e-5, atol=1e-5)
    assert torch.isfinite(lse[3]), lse[3]
    assert torch.isneginf(blank_lp[3]) and torch.isneginf(label_lp[3]), (blank_lp[3], label_lp[3])


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


def test_monotonic_rnnt_needs_max_frames():
    # The batch's own maximum would be a host read, so the caller passes a static bound.
    args = (
        torch.randn(8, 3),
        torch.ones((1, 1), dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )
    try:
        monotonic_rnnt_loss(*args, blank=0)
    except TypeError:
        pass
    else:
        raise AssertionError("expected max_frames to be required")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        globals()[sys.argv[1]]()
    else:
        for name, func in sorted(globals().items()):
            if name.startswith("test_"):
                print(f"-- {name}")
                func()
        print("all passed")
