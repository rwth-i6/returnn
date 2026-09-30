"""
Tests for the packed RNN-T loss, see :mod:`returnn.torch.util.rnnt`.
"""

from __future__ import annotations

import itertools
import sys
import unittest

import torch

import _setup_test_env  # noqa
from returnn.torch.util.monotonic_rnnt import cell_offsets
from returnn.torch.util.rnnt import rnnt_loss


def _brute_force(logits: torch.Tensor, labels, blank: int) -> float:
    """
    Sums over every alignment explicitly, as the definition of the loss.

    An alignment gives every label the frame it is emitted in, the frames never going back,
    and leaves every frame by a blank behind the labels emitted in it.

    :param logits: [T, U + 1, V] of one sequence
    :param labels: the U labels
    :param blank: blank index
    :return: the negative log likelihood
    """
    num_frames, num_prefix, _ = logits.shape
    num_labels = num_prefix - 1
    log_probs = torch.log_softmax(logits.double(), dim=-1)
    totals = []
    for emits in itertools.combinations_with_replacement(range(num_frames), num_labels):
        u = 0
        total = 0.0
        for t in range(num_frames):
            while u < num_labels and emits[u] == t:
                total += float(log_probs[t, u, labels[u]])
                u += 1
            total += float(log_probs[t, u, blank])
        totals.append(total)
    if not totals:
        return 0.0
    return -float(torch.logsumexp(torch.tensor(totals, dtype=torch.float64), dim=0))


def _pack(per_seq_logits):
    """
    :param per_seq_logits: one [T, U + 1, V] tensor per sequence
    :return: the packed [cells, V] tensor the loss takes, frame index outer
    """
    return torch.cat([x.reshape(-1, x.shape[-1]) for x in per_seq_logits], dim=0)


def _batch(cases, vocab: int, *, extra_label_columns: int = 0):
    """
    :param cases: (frames, labels) per sequence
    :param vocab: vocabulary size with blank at index 0
    :param extra_label_columns: columns the padded labels are wider than the longest sequence
    :return: (logits per sequence, labels per sequence, padded labels, frame lens, label lens)
    """
    per_seq, labels = [], []
    for num_frames, num_labels in cases:
        per_seq.append(torch.randn(num_frames, num_labels + 1, vocab))
        seq_labels = torch.randint(1, vocab, (num_labels,))
        if num_labels >= 2:
            seq_labels[1] = seq_labels[0]
        labels.append(seq_labels)
    width = max(u for _t, u in cases) + extra_label_columns
    labels_padded = torch.zeros(len(cases), width, dtype=torch.int32)
    for i, seq_labels in enumerate(labels):
        labels_padded[i, : len(seq_labels)] = seq_labels
    frame_lens = torch.tensor([t for t, _u in cases], dtype=torch.int32)
    label_lens = torch.tensor([u for _t, u in cases], dtype=torch.int32)
    return per_seq, labels, labels_padded, frame_lens, label_lens


def test_rnnt_matches_the_sum_over_alignments():
    """
    The last sequence has more labels than frames, which the monotonic loss cannot align and this one can,
    since several labels may share a frame.
    """
    torch.manual_seed(42)
    vocab, blank = 5, 0
    cases = [(4, 2), (5, 0), (3, 3), (6, 2), (2, 4), (1, 3)]
    per_seq, labels, labels_padded, frame_lens, label_lens = _batch(cases, vocab, extra_label_columns=3)
    want = [_brute_force(logits, seq_labels.tolist(), blank) for logits, seq_labels in zip(per_seq, labels)]

    got = rnnt_loss(_pack(per_seq), labels_padded, frame_lens, label_lens, blank=blank)
    torch.testing.assert_close(got.double(), torch.tensor(want, dtype=torch.float64), rtol=0, atol=1e-5)


def test_rnnt_runs_over_a_buffer_and_a_recursion_above_the_batch():
    """
    A captured step hands over a buffer of the declared capacity and the declared frames, both above what
    the batch holds, and the loss of the batch must not change with either.
    """
    torch.manual_seed(9)
    vocab, blank = 5, 0
    cases = [(4, 2), (3, 3), (2, 0)]
    per_seq, labels, labels_padded, frame_lens, label_lens = _batch(cases, vocab, extra_label_columns=2)
    want = [_brute_force(logits, seq_labels.tolist(), blank) for logits, seq_labels in zip(per_seq, labels)]

    packed = torch.cat([_pack(per_seq), torch.randn(11, vocab)])
    got = rnnt_loss(packed, labels_padded, frame_lens, label_lens, blank=blank, max_frames=9)
    torch.testing.assert_close(got.double(), torch.tensor(want, dtype=torch.float64), rtol=0, atol=1e-5)


def test_rnnt_gradient_matches_finite_differences():
    torch.manual_seed(7)
    vocab, blank = 4, 0
    num_frames, num_labels = 3, 2
    logits = torch.randn(num_frames, num_labels + 1, vocab, dtype=torch.float64, requires_grad=True)
    args = (
        torch.tensor([[1, 3]], dtype=torch.int32),
        torch.tensor([num_frames], dtype=torch.int32),
        torch.tensor([num_labels], dtype=torch.int32),
    )
    loss = rnnt_loss(logits.reshape(-1, vocab), *args, blank=blank)
    want = _brute_force(logits.detach(), [1, 3], blank)
    torch.testing.assert_close(loss, torch.tensor([want], dtype=torch.float64), rtol=0, atol=1e-9)
    torch.autograd.gradcheck(
        lambda x: rnnt_loss(x.reshape(-1, vocab), *args, blank=blank),
        (logits,),
        eps=1e-6,
        atol=1e-6,
    )


def test_rnnt_handles_degenerate_batches():
    torch.manual_seed(3)
    vocab, blank = 6, 0

    frame_lens = torch.tensor([0, 0], dtype=torch.int32)
    label_lens = torch.tensor([5, 7], dtype=torch.int32)
    got = rnnt_loss(
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
    got = rnnt_loss(logits, torch.zeros((2, 0), dtype=torch.int32), frame_lens, label_lens, blank=blank, max_frames=4)
    blank_lp = torch.log_softmax(logits.double(), dim=-1)[:, blank]
    want = torch.tensor([-float(blank_lp[:4].sum()), -float(blank_lp[4:].sum())])
    torch.testing.assert_close(got.double(), want.double(), rtol=0, atol=1e-5)

    frame_lens = torch.tensor([2, 0], dtype=torch.int32)
    label_lens = torch.tensor([1, 3], dtype=torch.int32)
    logits = torch.randn(4, vocab, requires_grad=True)
    labels = torch.randint(1, vocab, (2, 3), dtype=torch.int32)
    got = rnnt_loss(logits, labels, frame_lens, label_lens, blank=blank)
    want = _brute_force(logits.detach().reshape(2, 2, vocab), labels[0, :1].tolist(), blank)
    torch.testing.assert_close(got.double(), torch.tensor([want, 0.0], dtype=torch.float64), rtol=0, atol=1e-5)
    got.sum().backward()
    assert torch.isfinite(logits.grad).all(), logits.grad


def test_rnnt_refuses_a_recursion_shorter_than_a_sequence():
    torch.manual_seed(1)
    logits = torch.randn(12, 5)
    labels = torch.tensor([[1, 2]], dtype=torch.int32)
    frame_lens, label_lens = torch.tensor([4], dtype=torch.int32), torch.tensor([2], dtype=torch.int32)
    fine = rnnt_loss(logits, labels, frame_lens, label_lens, blank=0, max_frames=4)
    assert torch.isfinite(fine).all() and float(fine) > 0.0, fine

    try:
        rnnt_loss(logits, labels, frame_lens, label_lens, blank=0, max_frames=3)
    except RuntimeError as exc:
        assert "frames" in str(exc), exc
    else:
        raise AssertionError("a recursion shorter than the sequence was accepted")


def test_rnnt_refuses_labels_outside_the_vocabulary():
    torch.manual_seed(1)
    vocab = 5
    logits = torch.randn(12, vocab)
    frame_lens, label_lens = torch.tensor([4], dtype=torch.int32), torch.tensor([2], dtype=torch.int32)
    fine = rnnt_loss(logits, torch.tensor([[1, 2]], dtype=torch.int32), frame_lens, label_lens, blank=0)
    assert torch.isfinite(fine).all() and float(fine) > 0.0, fine

    for labels, blank, error in (
        ([[1, 2]], vocab, AssertionError),
        ([[1, vocab]], 0, RuntimeError),
        ([[-1, 2]], 0, RuntimeError),
    ):
        try:
            rnnt_loss(logits, torch.tensor(labels, dtype=torch.int32), frame_lens, label_lens, blank=blank)
        except error as exc:
            assert "vocabulary" in str(exc), exc
        else:
            raise AssertionError(f"labels {labels} with blank {blank} were accepted")


def test_rnnt_on_cuda_matches_the_reference():
    """
    What the kernels see under capture, a buffer and a recursion above the batch's own cells and frames,
    a sequence with more labels than frames, slots without frames and a batch without any label, each
    against the cpu path.
    """
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no cuda")
    torch.manual_seed(3)
    vocab, blank, slack = 37, 0, 13
    for cases in ([(9, 4), (12, 0), (7, 7), (5, 2), (3, 5), (1, 6), (0, 3)], [(4, 0), (6, 0)], [(1, 40), (30, 1)]):
        per_seq, _labels, labels_padded, frame_lens, label_lens = _batch(cases, vocab, extra_label_columns=2)
        packed = torch.cat([_pack(per_seq), torch.randn(slack, vocab)])
        grad_out = torch.randn(len(cases))

        results = []
        for device in ("cpu", "cuda"):
            logits = packed.to(device).clone().requires_grad_()
            loss = rnnt_loss(
                logits,
                labels_padded.to(device),
                frame_lens.to(device),
                label_lens.to(device),
                blank=blank,
                max_frames=int(frame_lens.max()) + 2,
            )
            loss.backward(grad_out.to(device))
            results.append((loss.detach().cpu(), logits.grad.detach().cpu()))
        torch.testing.assert_close(results[1][0], results[0][0], rtol=1e-5, atol=1e-5, msg=str(cases))
        torch.testing.assert_close(results[1][1], results[0][1], rtol=1e-4, atol=1e-5, msg=str(cases))


def test_rnnt_scans_on_cuda_give_the_posteriors_of_every_edge():
    """
    The edge posteriors of a frame's blanks sum to one, since every alignment leaves every frame exactly once,
    and those of all label edges sum to the number of labels.
    """
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no cuda")
    from returnn.torch.util.rnnt_triton import backward_scan, forward_scan

    torch.manual_seed(4)
    frame_lens = torch.tensor([3, 4, 0, 1], dtype=torch.int32, device="cuda")
    label_lens = torch.tensor([5, 2, 3, 0], dtype=torch.int32, device="cuda")
    offsets, cells = cell_offsets(frame_lens, label_lens)
    num_cells = int(cells.sum()) + 5
    log_probs = torch.log_softmax(torch.randn(num_cells, 3, device="cuda"), dim=-1)
    blank_lp, label_lp = log_probs[:, 0].contiguous(), log_probs[:, 1].contiguous()

    total, alpha = forward_scan(blank_lp, label_lp, offsets, frame_lens, label_lens, 6, 8)
    blank_post, label_post = backward_scan(
        blank_lp, label_lp, offsets, frame_lens, label_lens, alpha, total, torch.ones(4, device="cuda"), 6, 8
    )
    for seq in range(4):
        first, count = int(offsets[seq]), int(cells[seq])
        frames, prefixes = int(frame_lens[seq]), int(label_lens[seq]) + 1
        per_frame = blank_post[first : first + count].reshape(frames, prefixes).sum(dim=1)
        torch.testing.assert_close(per_frame, torch.ones(frames, device="cuda"), rtol=1e-4, atol=1e-4)
        labels = label_post[first : first + count].sum()
        want = float(prefixes - 1) if frames else 0.0
        torch.testing.assert_close(labels, torch.tensor(want, device="cuda"), rtol=1e-4, atol=1e-4)
    assert not blank_post[int(cells.sum()) :].any() and not label_post[int(cells.sum()) :].any()


def test_rnnt_traces_under_aot():
    if not torch.cuda.is_available():
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
        return rnnt_loss(x, lab, flens, ulens, blank=blank, max_frames=frames)

    want = []
    for func in (loss_of, aot_function(loss_of, fw_compiler=nop, bw_compiler=nop)):
        x = packed.clone().requires_grad_()
        loss = func(x, labels, frame_lens, label_lens)
        loss.backward(grad_out)
        want.append((loss.detach().clone(), x.grad.clone()))
    torch.testing.assert_close(want[1][0], want[0][0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(want[1][1], want[0][1], rtol=1e-4, atol=1e-6)


def test_rnnt_takes_the_logits_in_their_own_dtype():
    """
    The kernels read every logit anyway, so they convert on load and the gradient comes back in the dtype of
    the logits, without a float32 copy of the lattice in between.
    """
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no cuda")
    cells, vocab = 4096, 1024
    frame_lens = torch.tensor([32, 32], dtype=torch.int32, device="cuda")
    label_lens = torch.tensor([63, 63], dtype=torch.int32, device="cuda")
    labels = torch.randint(0, vocab - 1, (2, 63), dtype=torch.int32, device="cuda")
    logits = torch.randn(cells, vocab, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    loss = rnnt_loss(logits, labels, frame_lens, label_lens, blank=vocab - 1, max_frames=32)
    loss.sum().backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before

    assert logits.grad.dtype == torch.bfloat16, logits.grad.dtype
    lattice_fp32 = cells * vocab * 4
    assert peak < lattice_fp32, f"peak {peak} implies a float32 copy of the lattice ({lattice_fp32} bytes)"

    reference = logits.detach().float().requires_grad_()
    want = rnnt_loss(reference, labels, frame_lens, label_lens, blank=vocab - 1, max_frames=32)
    want.sum().backward()
    torch.testing.assert_close(loss, want, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(logits.grad.float(), reference.grad, rtol=2e-2, atol=2e-5)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        globals()[sys.argv[1]]()
    else:
        for name, func in sorted(globals().items()):
            if name.startswith("test_"):
                print(f"-- {name}")
                try:
                    func()
                except unittest.SkipTest as exc:
                    print(f"   (skipped, {exc})")
        print("all passed")
